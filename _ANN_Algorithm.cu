#include "_ANN_Algorithm.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <device_functions.h>


/******************************************************

					 Host Functions

*******************************************************/

__host__ cudaError_t _ANN_ProductMat(GMatrix input, GMatrix weight, GMatrix bias, GMatrix output){
	dim3 threads(32, 32);
	dim3 blocks((output.d1 + 31) / 32, (output.d2 + 31) / 32);

	GMat2D _input = {input.gmem2d, input.d1, input.d2};
	GMat2D _output = {output.gmem2d, output.d1, output.d2};

	_ANN_ProductMat_Kernel<<<blocks, threads>>>(_input, weight.gmem2d, bias.gmem1d, _output);

	return cudaGetLastError();
}

__host__ cudaError_t _ANN_ReLU(GMatrix io){
	dim3 threads(1024);
	dim3 blocks((io.t_elements + 1023) / 1024);

	GMat1D _io = {io.gmem1d, io.t_elements};

	_ANN_ReLU_Kernel<<<blocks, threads>>>(_io);

	return cudaGetLastError();
}

__host__ cudaError_t _SoftMax(GMatrix io){
	dim3 threads(32);
	dim3 blocks((io.d2 + 31) / 32);

	GMat2D _io = {io.gmem2d, io.d1, io.d2};

	_SoftMax_Kernel<<<blocks, threads>>>(_io);

	return cudaGetLastError();
}

__host__ cudaError_t _CrossEntropy(GMatrix target, GMatrix output, float *cost){
	dim3 threads(1024);
	dim3 blocks(1);

	GMat2D _output = {output.gmem2d, output.d1, output.d2};

	_CrossEntropy_kernel<<<blocks, threads>>>(target.gmem2d, _output, cost);

	return cudaGetLastError();
}

__host__ cudaError_t _CrossEntropyDelta(GMatrix target, GMatrix output, GMatrix delta){
	dim3 threads(32, 32);
	dim3 blocks((output.d1 + 31) / 32, (output.d2 + 31) / 32);

	GMat2D _output = {output.gmem2d, output.d1, output.d2};

	_CrossEntropyDelta_Kernel<<<blocks, threads>>>(target.gmem2d, _output, delta.gmem2d);

	return cudaGetLastError();
}

__host__ cudaError_t _ANN_D_ReLU(GMatrix ioDelta, GMatrix output){
	dim3 threads(1024);
	dim3 blocks((output.t_elements + 1023) / 1024);

	GMat1D _ioDelta = {ioDelta.gmem1d, ioDelta.t_elements};

	_ANN_D_ReLU_Kernel<<<blocks, threads>>>(_ioDelta, output.gmem1d);

	return cudaGetLastError();
}

__host__ cudaError_t _ANN_D_ProductMat(GMatrix inDelta, GMatrix weight, GMatrix outDelta){
	dim3 threads(32, 32);
	dim3 blocks((outDelta.d1 + 31) / 32, (outDelta.d2 + 31) / 32);

	GMat2D _inDelta = {inDelta.gmem2d, inDelta.d1, inDelta.d2};
	GMat2D _outDelta = {outDelta.gmem2d, outDelta.d1, outDelta.d2};

	_ANN_D_ProductMat_Kernel<<<blocks, threads>>>(_inDelta, weight.gmem2d, _outDelta);

	return cudaGetLastError();
}

__host__ cudaError_t _ANN_ModifyWeight(GMatrix delta, GMatrix weight, GMatrix input, GMatrix wMoment, float learnRate, float momentRate){
	dim3 threads(32, 32);
	dim3 blocks((weight.d1 + 31) / 32, (weight.d2 + 31) / 32);

	GMat2D _delta = {delta.gmem2d, delta.d1, delta.d2};
	GMat2D _input = {input.gmem2d, input.d1, input.d2};

	_ANN_ModifyWeight_kernel<<<blocks, threads>>>(_delta, weight.gmem2d, _input, wMoment.gmem2d, learnRate, momentRate);

	return cudaGetLastError();
}

__host__ cudaError_t _ANN_ModifyBias(GMatrix delta, GMatrix bias, GMatrix bMoment, float learnRate, float momentRate){
	dim3 threads(32);
	dim3 blocks((bias.d1 + 31) / 32);

	GMat2D _delta = {delta.gmem2d, delta.d1, delta.d2};

	_ANN_ModifyBias_Kernel<<<blocks, threads>>>(_delta, bias.gmem1d, bMoment.gmem1d, learnRate, momentRate);

	return cudaGetLastError();
}




/******************************************************

					Device Functions

*******************************************************/

__global__ void _ANN_ProductMat_Kernel(GMat2D input, float **weight, float *bias, GMat2D output){
	int om = blockDim.x * blockIdx.x + threadIdx.x; // output nodes
	int n = blockDim.y * blockIdx.y + threadIdx.y; // output samples;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	__shared__ float _input[32][32];
	__shared__ float _weight[32][32];
	float _sum = 0;

	for(int im = 0; im < input.d1; im += 32){
		if(tx + im < input.d1 && n < input.d2)
			_input[ty][tx] = input.data[n][tx + im];
		else
			_input[ty][tx] = 0.f;

		if(ty + im < input.d1 && om < output.d1)
			_weight[ty][tx] = weight[ty + im][om];
		else
			_weight[ty][tx] = 0;
		__syncthreads();

		for(int k = 0; k < 32; ++k){
			_sum += _input[ty][k] * _weight[k][tx];
		}
		__syncthreads();
	}

	if(om < output.d1 && n < output.d2) output.data[n][om] = _sum + bias[om];
}

__global__ void _ANN_ReLU_Kernel(GMat1D io){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < io.d1) io.data[idx] = __max(0.f, io.data[idx]);
}

__global__ void _SoftMax_Kernel(GMat2D io){
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if(n < io.d2){
		float _sum = 0.f;
		for(int i = 0; i < io.d1; ++i) _sum += io.data[n][i] = expf(io.data[n][i]);
		for(int i = 0; i < io.d1; ++i) io.data[n][i] /= _sum;
	}
}

__global__ void _CrossEntropy_kernel(float **target, GMat2D output, float *cost){
	__shared__ float _cost[1024];
	float _sum = 0.f;

	_cost[threadIdx.x] = 0.f;
	__syncthreads();

	for(int n = threadIdx.x; n < output.d2; n += 1024){
		for(int m = 0; m < output.d1; ++m){
			if(target[n][m]) _sum -= log(output.data[n][m]);
		}
	}
	_cost[threadIdx.x] = _sum;
	__syncthreads();

	if(threadIdx.x < 512) _cost[threadIdx.x] += _cost[threadIdx.x + 512]; __syncthreads();
	if(threadIdx.x < 256) _cost[threadIdx.x] += _cost[threadIdx.x + 256]; __syncthreads();
	if(threadIdx.x < 128) _cost[threadIdx.x] += _cost[threadIdx.x + 128]; __syncthreads();
	if(threadIdx.x < 64) _cost[threadIdx.x] += _cost[threadIdx.x + 64]; __syncthreads();
	if(threadIdx.x < 32) _cost[threadIdx.x] += _cost[threadIdx.x + 32]; __syncthreads();
	if(threadIdx.x < 16) _cost[threadIdx.x] += _cost[threadIdx.x + 16]; __syncthreads();
	if(threadIdx.x < 8) _cost[threadIdx.x] += _cost[threadIdx.x + 8]; __syncthreads();
	if(threadIdx.x < 4) _cost[threadIdx.x] += _cost[threadIdx.x + 4]; __syncthreads();
	if(threadIdx.x < 2) _cost[threadIdx.x] += _cost[threadIdx.x + 2]; __syncthreads();
	if(threadIdx.x < 1) _cost[threadIdx.x] += _cost[threadIdx.x + 1]; __syncthreads();
	if(threadIdx.x == 0) *cost = _cost[threadIdx.x] / output.d2;
}

__global__ void _CrossEntropyDelta_Kernel(float **target, GMat2D output, float **delta){
	int m = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;

	if(n < output.d2 && m < output.d1){
		delta[n][m] = target[n][m] - output.data[n][m];
	}
}

__global__ void _ANN_D_ReLU_Kernel(GMat1D ioDelta, float *output){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < ioDelta.d1){
		if(output[idx] == 0.f) ioDelta.data[idx] = 0.f;
	}
}

__global__ void _ANN_D_ProductMat_Kernel(GMat2D inDelta, float **weight, GMat2D outDelta){
	int im = blockDim.x * blockIdx.x + threadIdx.x;	// nInput
	int n = blockDim.y * blockIdx.y + threadIdx.y;	// nSample
	float _sum = 0;

	for(int om = 0; om < inDelta.d1; om += 32){
		__shared__ float _delta[32][32];
		__shared__ float _weight[32][32];
		
		if(threadIdx.x + om < inDelta.d1 && n < inDelta.d2)
			_delta[threadIdx.y][threadIdx.x] = inDelta.data[n][threadIdx.x + om];
		else
			_delta[threadIdx.y][threadIdx.x] = 0;

		if(im < outDelta.d1 && threadIdx.y + om < inDelta.d1)
			_weight[threadIdx.y][threadIdx.x] = weight[im][threadIdx.y + om];
		else
			_weight[threadIdx.y][threadIdx.x] = 0;
		
		__syncthreads();

		for(int e = 0; e < 32; ++e){
			_sum += _delta[threadIdx.y][e] * _weight[e][threadIdx.x];
		}
		__syncthreads();
	}
	if(im < outDelta.d1 && n < outDelta.d2) outDelta.data[n][im] = _sum;
}

__global__ void _ANN_ModifyWeight_kernel(GMat2D delta, float **weight, GMat2D input, float **wMoment, float learnRate, float momentRate){
	int om = blockDim.x * blockIdx.x + threadIdx.x; // output
	int im = blockDim.y * blockIdx.y + threadIdx.y;	// input
	float _sum = 0;
	
	for(int n = 0; n < input.d2; n += 32){
		__shared__ float _input[32][32];
		__shared__ float _delta[32][32];

		if(threadIdx.x + n < input.d2 && im < input.d1)
			_input[threadIdx.y][threadIdx.x] = input.data[threadIdx.x + n][im];
		else 
			_input[threadIdx.y][threadIdx.x] = 0;

		if(om < delta.d1 && threadIdx.y + n < delta.d2)
			_delta[threadIdx.y][threadIdx.x] = delta.data[threadIdx.y + n][om];
		else
			_delta[threadIdx.y][threadIdx.x] = 0;
		__syncthreads();

		for(int e = 0; e < 32; ++e){
			_sum += _input[threadIdx.y][e] * _delta[e][threadIdx.x];
		}
		__syncthreads();
	}
	if(om < delta.d1 && im < input.d1){
		float val = _sum / input.d2;
		float G = wMoment[im][om] = momentRate * wMoment[im][om] + (1 - momentRate) * val * val;
		weight[im][om] += (learnRate / sqrt(G + 0.000001f)) * val;
	}
	__syncthreads();
}

__global__ void _ANN_ModifyBias_Kernel(GMat2D delta, float *bias, float *bMoment, float learnRate, float momentRate){
	int om = blockDim.x * blockIdx.x + threadIdx.x;

	if(om < delta.d1){
		float _sum = 0;

		for(int n = 0; n < delta.d2; ++n){
			_sum += delta.data[n][om];
		}
		float val = _sum / delta.d2;
		float G = bMoment[om] = momentRate * bMoment[om] + (1 - momentRate) * val * val;
		bias[om] += (learnRate / sqrt(G + 0.000001f)) * val;
	}
}