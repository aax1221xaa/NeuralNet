#ifndef _CNN_ALGORITHM_CUH
#define _CNN_ALGORITHM_CUH

#include "NN_Misc.h"



/******************************************************

					 Host Functions

*******************************************************/

__host__ cudaError_t _CNN_Convolution(GMatrix input, int stride, int padding, GMatrix kernel, GMatrix bias, GMatrix output);
__host__ cudaError_t _CNN_ReLU(GMatrix io);
__host__ cudaError_t _CNN_MaxPool(GMatrix input, GMark mark, int poolSize, GMatrix output);
__host__ cudaError_t _CNN_UnPool(GMatrix inDelta, GMark mark, int poolSize, GMatrix outDelta);
__host__ cudaError_t _CNN_D_ReLU(GMatrix inDelta, GMatrix output, int stride, GMatrix outDelta);
__host__ cudaError_t _CNN_Correlation(GMatrix inDelta, GMatrix kernel, int padding, GMatrix outDelta);
__host__ cudaError_t _CNN_ModifyKernel(GMatrix delta, GMatrix kernel, GMatrix input, int padding, GMatrix kMoment, float learnRate, float momentRate);
__host__ cudaError_t _CNN_ModifyBias(GMatrix delta, GMatrix bias, GMatrix bMoment, float learnRate, float momentRate);



/******************************************************

					Device Functions

*******************************************************/

__global__ void _CNN_Convolution_Kernel(GMat4D input, int stride, int padding, GMat4D kernel, float *bias, GMat4D output);
__global__ void _CNN_ReLU_Kernel(GMat1D io);
__global__ void _CNN_MaxPool_Kernel(float ***input, int ***mark, int poolSize, GMat3D output);
__global__ void _CNN_UnPool_Kernel(GMat3D inDelta, int ***mark, int poolSize, float ***outDelta);
__global__ void _CNN_D_ReLU_Kernel(GMat3D inDelta, float ***output, int stride, float ***outDelta);
__global__ void _CNN_Correlation_Kernel(GMat4D inDelta, GMat4D kernel, int padding, GMat4D outDelta);
__global__ void _CNN_ModifyKernel_Kernel(GMat4D delta, GMat4D kernel, GMat4D input, int padding, float ****kMoment, float learnRate, float momentRate);
__global__ void _CNN_ModifyBias_Kernel(GMat4D delta, float *bias, float *bMoment, float learnRate, float momentRate);

#endif