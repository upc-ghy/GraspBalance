
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_gpu.h"


__global__ void group_points_grad_kernel_fast(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ grad_out, const int *__restrict__ idx, float *__restrict__ grad_points) {
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;
    grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx; 
    
    atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0] , grad_out[0]);
}

void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *grad_out, const int *idx, float *grad_points) {
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, nsample, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel_fast(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;

    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx; 
    int in_idx = bs_idx * c * n + c_idx * n + idx[0];
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}


void group_points_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out) {
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, nsample, points, idx, out);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
