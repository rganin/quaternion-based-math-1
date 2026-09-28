"""Quaternion-valued linear algebra for Transformer compression research."""

from .attention import (
    quaternion_component_attention,
    quaternion_conjugate,
    shared_score_attention,
    standard_attention,
)
from .factorized import FactorizedQuaternionLinear, parameter_count
from .linear import QuaternionLinear, quaternion_linear
from .lora import LoRAQuaternionLinear, inject_lora, lora_parameters
from .metrics import (
    cosine_similarity,
    max_abs_error,
    mse,
    reconstruction_report,
    relative_frobenius_error,
    relative_l2_error,
    rmse,
)
from .packing import (
    build_block_matrix,
    check_divisible,
    pack_components,
    pack_real_to_quaternion,
    split_components,
    unpack_quaternion_to_real,
)
from .planar import quaternion_linear_naive, quaternion_linear_planar
from .reference import hamilton_product, quaternion_linear_reference
from .unit import UnitQuaternionLinear, normalize_quaternion

__all__ = [
    # core
    "QuaternionLinear",
    "quaternion_linear",
    "quaternion_linear_reference",
    "hamilton_product",
    "pack_real_to_quaternion",
    "unpack_quaternion_to_real",
    "split_components",
    "pack_components",
    "build_block_matrix",
    "check_divisible",
    # planar
    "quaternion_linear_naive",
    "quaternion_linear_planar",
    # factorized
    "FactorizedQuaternionLinear",
    "parameter_count",
    # attention
    "standard_attention",
    "quaternion_component_attention",
    "shared_score_attention",
    "quaternion_conjugate",
    # unit quaternion
    "UnitQuaternionLinear",
    "normalize_quaternion",
    # lora
    "LoRAQuaternionLinear",
    "inject_lora",
    "lora_parameters",
    # metrics
    "mse",
    "rmse",
    "relative_frobenius_error",
    "relative_l2_error",
    "cosine_similarity",
    "max_abs_error",
    "reconstruction_report",
]
