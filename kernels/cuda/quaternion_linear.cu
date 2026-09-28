// Fused quaternion linear CUDA kernel (plan.md section 45, K4).
//
// Custom hand-written kernel: each block computes a [BM, 4*BN] output tile
// (all four quaternion components), looping over input groups once and reusing
// shared-memory activation and weight tiles across the four Hamilton equations.
// The real block matrix is never materialized.
//
// This is a tensor-core-free (SIMT) fused kernel and is intended as the
// hand-written CUDA reference point; a CUTLASS/CuTe tensor-core implementation
// remains future work.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T, int BM, int BN, int BK>
__global__ void quat_kernel(const T* __restrict__ X,
                            const T* __restrict__ WR, const T* __restrict__ WI,
                            const T* __restrict__ WJ, const T* __restrict__ WK,
                            T* __restrict__ Y,
                            int M, int K_G, int N_G) {
  __shared__ float xs[4][BM][BK];
  __shared__ float ws[4][BK][BN];

  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const int tx = threadIdx.x;  // 0..BN-1
  const int ty = threadIdx.y;  // 0..BM-1
  const int nthreads = blockDim.x * blockDim.y;
  const int tid = ty * blockDim.x + tx;

  float ar = 0.f, ai = 0.f, aj = 0.f, ak = 0.f;

  for (int k0 = 0; k0 < K_G; k0 += BK) {
    // Load activations: X[m, 4*kg + c]
    for (int idx = tid; idx < BM * BK; idx += nthreads) {
      const int i = idx / BK;
      const int k = idx % BK;
      const int m = m0 + i;
      const int kg = k0 + k;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      if (m < M && kg < K_G) {
        const T* base = X + (size_t)m * (4 * K_G) + (size_t)4 * kg;
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = to_float<T>(base[c]);
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) xs[c][i][k] = v[c];
    }
    // Load weights: Wc[kg, n] (packed [K_G, N_G] layout)
    for (int idx = tid; idx < BK * BN; idx += nthreads) {
      const int k = idx / BN;
      const int j = idx % BN;
      const int kg = k0 + k;
      const int n = n0 + j;
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      if (kg < K_G && n < N_G) {
        v[0] = to_float<T>(WR[(size_t)kg * N_G + n]);
        v[1] = to_float<T>(WI[(size_t)kg * N_G + n]);
        v[2] = to_float<T>(WJ[(size_t)kg * N_G + n]);
        v[3] = to_float<T>(WK[(size_t)kg * N_G + n]);
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) ws[c][k][j] = v[c];
    }
    __syncthreads();

    if (tx < BN && ty < BM) {
#pragma unroll 4
      for (int k = 0; k < BK; ++k) {
        const float xr = xs[0][ty][k], xi = xs[1][ty][k];
        const float xj = xs[2][ty][k], xk = xs[3][ty][k];
        const float wr = ws[0][k][tx], wi = ws[1][k][tx];
        const float wj = ws[2][k][tx], wk = ws[3][k][tx];
        ar += xr * wr - xi * wi - xj * wj - xk * wk;
        ai += xr * wi + xi * wr - xj * wk + xk * wj;
        aj += xr * wj + xi * wk + xj * wr - xk * wi;
        ak += xr * wk - xi * wj + xj * wi + xk * wr;
      }
    }
    __syncthreads();
  }

  if (tx < BN && ty < BM) {
    const int m = m0 + ty;
    const int n = n0 + tx;
    if (m < M && n < N_G) {
      T* out = Y + (size_t)m * (4 * N_G) + (size_t)4 * n;
      out[0] = from_float<T>(ar);
      out[1] = from_float<T>(ai);
      out[2] = from_float<T>(aj);
      out[3] = from_float<T>(ak);
    }
  }
}

torch::Tensor quaternion_linear_forward(torch::Tensor x,
                                        torch::Tensor wr, torch::Tensor wi,
                                        torch::Tensor wj, torch::Tensor wk) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  auto xc = x.contiguous();
  const int64_t K = xc.size(-1);
  TORCH_CHECK(K % 4 == 0, "in_features must be divisible by 4");
  const int64_t K_G = K / 4;
  const int64_t N_G = wr.size(1);
  const int64_t N = N_G * 4;
  TORCH_CHECK(wr.size(0) == K_G, "packed weight in_groups must match input");

  auto xf = xc.reshape({-1, K});
  const int64_t M = xf.size(0);
  auto y = torch::empty({M, N}, xf.options());

  constexpr int BM = 16, BN = 16, BK = 32;
  dim3 grid((M + BM - 1) / BM, (N_G + BN - 1) / BN);
  dim3 block(BN, BM);

  if (xf.scalar_type() == at::kFloat) {
    quat_kernel<float, BM, BN, BK><<<grid, block>>>(
        xf.data_ptr<float>(), wr.data_ptr<float>(), wi.data_ptr<float>(),
        wj.data_ptr<float>(), wk.data_ptr<float>(), y.data_ptr<float>(),
        (int)M, (int)K_G, (int)N_G);
  } else if (xf.scalar_type() == at::kBFloat16) {
    quat_kernel<__nv_bfloat16, BM, BN, BK><<<grid, block>>>(
        reinterpret_cast<__nv_bfloat16*>(xf.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(wr.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(wi.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(wj.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(wk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)K_G, (int)N_G);
  } else {
    TORCH_CHECK(false, "only float32 and bfloat16 are supported");
  }

  auto lead = x.sizes().vec();
  lead.back() = N;
  return y.reshape(lead);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quaternion_linear", &quaternion_linear_forward,
        "Fused quaternion linear (CUDA)");
}
