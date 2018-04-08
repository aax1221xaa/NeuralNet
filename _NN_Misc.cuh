#ifndef _NN_MISC_CUH
#define _NN_MISC_CUH


#include "cuda_runtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>



cudaError_t SetGMat2D(float **dst, float *src, int iter, int step);
cudaError_t SetGMat3D(float ***dst, float **src, int iter, int step);
cudaError_t SetGMat4D(float ****dst, float ***src, int iter, int step);

cudaError_t SetGMark2D(int **dst, int *src, int iter, int step);
cudaError_t SetGMark3D(int ***dst, int **src, int iter, int step);
cudaError_t SetGMark4D(int ****dst, int ***src, int iter, int step);

#endif