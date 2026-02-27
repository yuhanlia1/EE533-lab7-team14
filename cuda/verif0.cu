#include <stdint.h>
#include <cuda_bf16.h> // 必须包含此头文件以支持 BFloat16 [cite: 44]

// 1. 整数向量加法 (操作 int16_t 数据) [cite: 39, 41]
__global__ void vec_add_int16(int16_t *a, int16_t *b, int16_t *c) {
    // 获取当前线程的 ID [cite: 32]
    int idx = threadIdx.x;
    
    // 执行逐元素的加法 [cite: 41]
    c[idx] = a[idx] + b[idx];
}

// 2. BFloat16 融合乘加运算 (FMA: d = a * b + c) [cite: 49]
__global__ void bf16_fma(__nv_bfloat16 *a, __nv_bfloat16 *b, __nv_bfloat16 *c, __nv_bfloat16 *d) {
    // 获取当前线程的 ID [cite: 32]
    int idx = threadIdx.x;
    
    // 使用 CUDA 内置的 BFloat16 FMA 函数 [cite: 54]
    d[idx] = __hfma(a[idx], b[idx], c[idx]);
}
