import torch
import os
import triton
import triton.language as tl
from menlo_kernels import triton_rowscaled_mm
from typing import Any, Optional
from vllm.model_executor.layers.quantization.utils.w8a8_utils import maybe_create_device_identity
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.linear import LinearMethodBase, LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.parameter import ModelWeightParameter, Parameter, ChannelQuantScaleParameter

os.environ['TRITON_ALWAYS_COMPILE'] = '1'

class SmallMTRITONConfig(QuantizationConfig):
    """Config class for SmallM TRITON"""

    def __init__(self,
                 weight_dtype: str = "int8",
                 lm_head_quantized: bool = False) -> None:
        super().__init__()
        self.weight_dtype = weight_dtype  # "int8" or "fp8"
        # rowscaled kernel uses no grouping along K
        self.group_size = -1
        self.lm_head_quantized = lm_head_quantized

    def __repr__(self) -> str:
        return (f"SmallMTRITONConfig(weight_dtype={self.weight_dtype}, "
                f"group_size={self.group_size}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls):
        return "small_m_triton"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton kernels here target Ampere+ by default
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        # No specific on-disk config format is required
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SmallMTRITONConfig":
        # Accept optional fields; default to int8 weights
        weight_dtype = getattr(cls, "get_from_keys_or", None)
        if weight_dtype is None:
            # fallback if helper not present in this build
            weight_dtype_val = config.get("weight_dtype", "int8")
            lm_head_quantized = bool(config.get("lm_head", False))
        else:
            weight_dtype_val = cls.get_from_keys_or(config, ["weight_dtype"],
                                                    "int8")
            lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                     False)
        return cls(weight_dtype=weight_dtype_val,
                   lm_head_quantized=lm_head_quantized)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return SmallMTRITONLinearMethod(self)
        return None


class SmallMTRITONLinearMethod(LinearMethodBase):
    """Linear method for SmallM TRITON.

    Args:
        quant_config: The SmallM TRITON quantization config.
    """

    def __init__(self, quant_config: QuantizationConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Normalize group_size
        assert self.quant_config.group_size == -1, "We do not support group_size for SmallM TRITON"
        # WEIGHT
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        
        # Initialise weights
        weight_dtype = params_dtype # Assume no serialized fp8 for now

        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn if self.quant_config.weight_dtype == "fp8" else torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=extra_weight_attrs.get("weight_loader"))

        scale = ChannelQuantScaleParameter(
            output_dim=0,
            data=torch.empty(output_size_per_partition, 1,
                            dtype=params_dtype, device=weight.device),
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)
        
    def scale_inputs(self, tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Scale input and apply triton kernel
        absmax = tensor.abs().amax(dim=-1, keepdim=True)
        
        if self.quant_config.weight_dtype == "int8":
            scale = absmax / 127.5  # 127 should be fine too
            xq = (tensor / scale.clip(1e-4)).round().clip(-128, 127).to(torch.int8)

        elif self.quant_config.weight_dtype == "fp8":
            scale = absmax / 448.0
            xq = (tensor / scale.clip(1e-4)).clip(-448.0, 448.0).to(torch.float8_e4m3fn)

        return xq.to(device=device), scale.to(device=device)
        

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Scale weights
        layer.weight = Parameter(layer.weight.data.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data.t(), requires_grad=False)
        layer.input_scale = None


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Scale input and apply triton kernel
        xq, scale = self.scale_inputs(x, x.device)
        return triton_rowscaled_mm(
            A=xq,
            B=layer.weight,
            sA=scale,
            sB=layer.weight_scale,
        )
        