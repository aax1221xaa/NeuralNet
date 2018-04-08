#include "_NN_Misc.cuh"
#include "device_launch_parameters.h"
#include <device_functions.h>

template <typename _T>
__global__ void JoinMemBlock(_T *dst, _T src, int iter, int step){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx < iter) dst[idx] = src + idx * step;
}


cudaError_t SetGMat2D(float **dst, float *src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);

	return cudaGetLastError();
}

cudaError_t SetGMat3D(float ***dst, float **src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);

	return cudaGetLastError();
}

cudaError_t SetGMat4D(float ****dst, float ***src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);
	
	return cudaGetLastError();
}

cudaError_t SetGMark2D(int **dst, int *src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);

	return cudaGetLastError();
}

cudaError_t SetGMark3D(int ***dst, int **src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);

	return cudaGetLastError();
}

cudaError_t SetGMark4D(int ****dst, int ***src, int iter, int step){
	dim3 threads(1024);
	dim3 blocks((iter + 1023) / 1024);

	JoinMemBlock<<<blocks, threads>>>(dst, src, iter, step);

	return cudaGetLastError();
}