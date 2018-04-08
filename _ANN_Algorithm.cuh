#ifndef _ANN_ALGORITHM_CUH
#define _ANN_ALGORITHM_CUH

#include "NN_Misc.h"


/******************************************************

					 Host Functions

*******************************************************/

__host__ cudaError_t _ANN_ProductMat(GMatrix input, GMatrix weight, GMatrix bias, GMatrix output);
__host__ cudaError_t _ANN_ReLU(GMatrix io);
__host__ cudaError_t _SoftMax(GMatrix io);
__host__ cudaError_t _CrossEntropy(GMatrix target, GMatrix output, float *cost);
__host__ cudaError_t _CrossEntropyDelta(GMatrix target, GMatrix output, GMatrix delta);
__host__ cudaError_t _ANN_D_ReLU(GMatrix ioDelta, GMatrix output);
__host__ cudaError_t _ANN_D_ProductMat(GMatrix inDelta, GMatrix weight, GMatrix outDelta);
__host__ cudaError_t _ANN_ModifyWeight(GMatrix delta, GMatrix weight, GMatrix input, GMatrix wMoment, float learnRate, float momentRate);
__host__ cudaError_t _ANN_ModifyBias(GMatrix delta, GMatrix bias, GMatrix bMoment, float learnRate, float momentRate);



/******************************************************

					Device Functions

*******************************************************/

__global__ void _ANN_ProductMat_Kernel(GMat2D input, float **weight, float *bias, GMat2D output);
__global__ void _ANN_ReLU_Kernel(GMat1D io);
__global__ void _SoftMax_Kernel(GMat2D io);
__global__ void _CrossEntropy_kernel(float **target, GMat2D output, float *cost);
__global__ void _CrossEntropyDelta_Kernel(float **target, GMat2D output, float **delta);
__global__ void _ANN_D_ReLU_Kernel(GMat1D ioDelta, float *output);
__global__ void _ANN_D_ProductMat_Kernel(GMat2D inDelta, float **weight, GMat2D outDelta);
__global__ void _ANN_ModifyWeight_kernel(GMat2D delta, float **weight, GMat2D input, float **wMoment, float learnRate, float momentRate);
__global__ void _ANN_ModifyBias_Kernel(GMat2D delta, float *bias, float *bMoment, float learnRate, float momentRate);

#endif