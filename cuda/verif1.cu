// kernel.cu
#include <stdint.h>
#include <cuda_bf16.h>

struct __align__(8) i16x4 {
  int16_t x0, x1, x2, x3;
};

struct __align__(8) bf16x4 {
  __nv_bfloat16 x0, x1, x2, x3;
};

static __device__ __forceinline__ int16_t relu_i16(int16_t v) {
  return (v < 0) ? (int16_t)0 : v;
}

static __device__ __forceinline__ __nv_bfloat16 bf16_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  float fa = __bfloat162float(a);
  float fb = __bfloat162float(b);
  return __float2bfloat16_rn(fa * fb);
}

static __device__ __forceinline__ __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  float fa = __bfloat162float(a);
  float fb = __bfloat162float(b);
  float fc = __bfloat162float(c);
  return __float2bfloat16_rn(fa * fb + fc);
}

extern "C" __global__ void vec_add_i16x4(const i16x4* a, const i16x4* b, i16x4* out, int n_packed) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n_packed) {
    i16x4 va = a[tid];
    i16x4 vb = b[tid];
    i16x4 r;
    r.x0 = (int16_t)(va.x0 + vb.x0);
    r.x1 = (int16_t)(va.x1 + vb.x1);
    r.x2 = (int16_t)(va.x2 + vb.x2);
    r.x3 = (int16_t)(va.x3 + vb.x3);
    out[tid] = r;
  }
}

extern "C" __global__ void vec_sub_i16x4(const i16x4* a, const i16x4* b, i16x4* out, int n_packed) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n_packed) {
    i16x4 va = a[tid];
    i16x4 vb = b[tid];
    i16x4 r;
    r.x0 = (int16_t)(va.x0 - vb.x0);
    r.x1 = (int16_t)(va.x1 - vb.x1);
    r.x2 = (int16_t)(va.x2 - vb.x2);
    r.x3 = (int16_t)(va.x3 - vb.x3);
    out[tid] = r;
  }
}

extern "C" __global__ void relu_i16x4(const i16x4* in, i16x4* out, int n_packed) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n_packed) {
    i16x4 v = in[tid];
    i16x4 r;
    r.x0 = relu_i16(v.x0);
    r.x1 = relu_i16(v.x1);
    r.x2 = relu_i16(v.x2);
    r.x3 = relu_i16(v.x3);
    out[tid] = r;
  }
}

extern "C" __global__ void vec_mul_bf16x4(const bf16x4* a, const bf16x4* b, bf16x4* out, int n_packed) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n_packed) {
    bf16x4 va = a[tid];
    bf16x4 vb = b[tid];
    bf16x4 r;
    r.x0 = bf16_mul(va.x0, vb.x0);
    r.x1 = bf16_mul(va.x1, vb.x1);
    r.x2 = bf16_mul(va.x2, vb.x2);
    r.x3 = bf16_mul(va.x3, vb.x3);
    out[tid] = r;
  }
}

extern "C" __global__ void fma_bf16x4(const bf16x4* a, const bf16x4* b, const bf16x4* c, bf16x4* out, int n_packed) {
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < n_packed) {
    bf16x4 va = a[tid];
    bf16x4 vb = b[tid];
    bf16x4 vc = c[tid];
    bf16x4 r;
    r.x0 = bf16_fma(va.x0, vb.x0, vc.x0);
    r.x1 = bf16_fma(va.x1, vb.x1, vc.x1);
    r.x2 = bf16_fma(va.x2, vb.x2, vc.x2);
    r.x3 = bf16_fma(va.x3, vb.x3, vc.x3);
    out[tid] = r;
  }
}
