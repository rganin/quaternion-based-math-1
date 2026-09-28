"""Custom fused CUDA quaternion kernel (plan.md section 45, K4).

A hand-written, tensor-core-free fused kernel.  A CUTLASS/CuTe tensor-core
implementation remains future work; see :mod:`kernels.cutlass`.
"""

from .loader import cuda_kernel_available, quaternion_linear_cuda

__all__ = ["cuda_kernel_available", "quaternion_linear_cuda"]
