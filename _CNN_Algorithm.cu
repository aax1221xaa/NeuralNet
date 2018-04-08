#include "_CNN_Algorithm.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include <device_functions.h>

/******************************************************

					 Host Functions

*******************************************************/

__host__ cudaError_t _CNN_Convolution(GMatrix input, int stride, int padding, GMatrix kernel, GMatrix bias, GMatrix output){
	dim3 threads(16, 16);
	dim3 blocks((output.d1 + 15) / 16, (output.d2 + 15) / 16, output.d3 * output.d4);

	GMat4D _input = {input.gmem4d, input.d1, input.d2, input.d3, input.d4};
	GMat4D _kernel = {kernel.gmem4d, kernel.d1, kernel.d2, kernel.d3, kernel.d4};
	GMat4D _output = {output.gmem4d, output.d1, output.d2, output.d3, output.d4};

	_CNN_Convolution_Kernel<<<blocks, threads>>>(_input, stride, padding, _kernel, bias.gmem1d, _output);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_ReLU(GMatrix io){
	dim3 threads(1024);
	dim3 blocks((io.t_elements + 1023) / 1024);

	GMat1D _io = {io.gmem1d, io.t_elements};

	_CNN_ReLU_Kernel<<<blocks, threads>>>(_io);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_MaxPool(GMatrix input, GMark mark, int poolSize, GMatrix output){
	dim3 threads(32, 32);
	dim3 blocks((output.d1 + 31) / 32, (output.d2 + 31) / 32, output.d3 * output.d4);

	GMat3D _output = {output.gmem3d, output.d1, output.d2, output.d3 * output.d4};

	_CNN_MaxPool_Kernel<<<blocks, threads>>>(input.gmem3d, mark.gmem3d, poolSize, _output);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_UnPool(GMatrix inDelta, GMark mark, int poolSize, GMatrix outDelta){
	dim3 threads(32, 32);
	dim3 blocks((inDelta.d1 + 31) / 32, (inDelta.d2 + 31) / 32, inDelta.d3 * inDelta.d4);

	GMat3D _inDelta = {inDelta.gmem3d, inDelta.d1, inDelta.d2, inDelta.d3 * inDelta.d4};

	_CNN_UnPool_Kernel<<<blocks, threads>>>(_inDelta, mark.gmem3d, poolSize, outDelta.gmem3d);
	
	return cudaGetLastError();
}

__host__ cudaError_t _CNN_D_ReLU(GMatrix inDelta, GMatrix output, int stride, GMatrix outDelta){
	dim3 threads(32, 32);
	dim3 blocks((inDelta.d1 + 31) / 32, (inDelta.d2 + 31) / 32, inDelta.d3 * inDelta.d4);

	GMat3D _inDelta = {inDelta.gmem3d, inDelta.d1, inDelta.d2, inDelta.d3 * inDelta.d4};

	_CNN_D_ReLU_Kernel<<<blocks, threads>>>(_inDelta, output.gmem3d, stride, outDelta.gmem3d);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_Correlation(GMatrix inDelta, GMatrix kernel, int padding, GMatrix outDelta){
	GMat4D _inDelta = {inDelta.gmem4d, inDelta.d1, inDelta.d2, inDelta.d3, inDelta.d4};
	GMat4D _kernel = {kernel.gmem4d, kernel.d1, kernel.d2, kernel.d3, kernel.d4};
	GMat4D _outDelta = {outDelta.gmem4d, outDelta.d1, outDelta.d2, outDelta.d3, outDelta.d4};

	dim3 threads(32, 32);
	dim3 blocks((outDelta.d1 + 31) / 32, (outDelta.d2 + 31) / 32, outDelta.d3 * outDelta.d4);
	_CNN_Correlation_Kernel<<<blocks, threads>>>(_inDelta, _kernel, padding, _outDelta);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_ModifyKernel(GMatrix delta, GMatrix kernel, GMatrix input, int padding, GMatrix kMoment, float learnRate, float momentRate){
	dim3 threads(32, 32);
	dim3 blocks(kernel.d1, kernel.d2, kernel.d3 * kernel.d4);

	GMat4D _delta = {delta.gmem4d, delta.d1, delta.d2, delta.d3, delta.d4};
	GMat4D _kernel = {kernel.gmem4d, kernel.d1, kernel.d2, kernel.d3, kernel.d4};
	GMat4D _input = {input.gmem4d, input.d1, input.d2, input.d3, input.d4};

	_CNN_ModifyKernel_Kernel<<<blocks, threads>>>(_delta, _kernel, _input, padding, kMoment.gmem4d, learnRate, momentRate);

	return cudaGetLastError();
}

__host__ cudaError_t _CNN_ModifyBias(GMatrix delta, GMatrix bias, GMatrix bMoment, float learnRate, float momentRate){
	dim3 threads(32, 32);
	dim3 blocks(delta.d3);

	GMat4D _delta = {delta.gmem4d, delta.d1, delta.d2, delta.d3, delta.d4};

	_CNN_ModifyBias_Kernel<<<blocks, threads>>>(_delta, bias.gmem1d, bMoment.gmem1d, learnRate, momentRate);
	
	return cudaGetLastError();
}




/******************************************************

					Device Functions

*******************************************************/

__global__ void _CNN_Convolution_Kernel(GMat4D input, int stride, int padding, GMat4D kernel, float *bias, GMat4D output){
	const int cx = blockDim.x * blockIdx.x + threadIdx.x;
	const int cy = blockDim.y * blockIdx.y + threadIdx.y;
	const int x_in = 15 * stride + kernel.d1;
	const int y_in = 15 * stride + kernel.d2;
	const int om = blockIdx.z % output.d3;
	const int nSample = blockIdx.z / output.d3;

	__shared__ float _input[56][56];
	float _sum = bias[om];
	// (in - k) / st + 1 = out
	// (out - 1) * st + k = in
	// (16 - 1) * 3 + 11 = 56

	for(int im = 0; im < input.d3; ++im){
		for(int y = threadIdx.y, _y = threadIdx.y + blockDim.y * blockIdx.y * stride - padding; y < y_in && _y < input.d2 + padding; y += 16, _y += 16){
			for(int x = threadIdx.x, _x = threadIdx.x + blockDim.x * blockIdx.x * stride - padding; x < x_in && _x < input.d1 + padding; x += 16, _x += 16){
				if(_y >= 0 && _y < input.d2 && _x >= 0 && _x < input.d1)
					_input[y][x] = input.data[nSample][im][_y][_x];
				else
					_input[y][x] = 0.f;
			}
		}
		__syncthreads();

		if(cx < output.d1 && cy < output.d2){
			for(int h = 0; h < kernel.d2; ++h){
				for(int w = 0; w < kernel.d1; ++w){
					_sum += _input[threadIdx.y * stride + h][threadIdx.x * stride + w] * kernel.data[im][om][h][w]; // __kernel[h * kSize.x + w];
				}
			}
		}
		__syncthreads();
	}
	if(cx < output.d1 && cy < output.d2) output.data[nSample][om][cy][cx] = _sum;
}

__global__ void _CNN_ReLU_Kernel(GMat1D io){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < io.d1) io.data[idx] = __max(0, io.data[idx]);
}

__global__ void _CNN_MaxPool_Kernel(float ***input, int ***mark, int poolSize, GMat3D output){
	int cy = blockDim.y * blockIdx.y + threadIdx.y;
	int cx = blockDim.x * blockIdx.x + threadIdx.x;

	if(cy < output.d2 && cx < output.d1){
		float _max = -FLT_MAX;
		int _mark = 0;

		for(int h = 0; h < poolSize; ++h){
			for(int w = 0; w < poolSize; ++w){
				if(_max < input[blockIdx.z][cy * poolSize + h][cx * poolSize + w]){
					_max = input[blockIdx.z][cy * poolSize + h][cx * poolSize + w];
					_mark = h * poolSize + w;
				}
			}
		}
		output.data[blockIdx.z][cy][cx] = _max;
		mark[blockIdx.z][cy][cx] = _mark;
	}
}

__global__ void _CNN_UnPool_Kernel(GMat3D inDelta, int ***mark, int poolSize, float ***outDelta){
	int cx = blockDim.x * blockIdx.x + threadIdx.x;
	int cy = blockDim.y * blockIdx.y + threadIdx.y;

	if(cy < inDelta.d2 && cx < inDelta.d1){
		for(int h = 0; h < poolSize; ++h){
			for(int w = 0; w < poolSize; ++w){
				if(h * poolSize + w == mark[blockIdx.z][cy][cx])
					outDelta[blockIdx.z][cy * poolSize + h][cx * poolSize + w] = inDelta.data[blockIdx.z][cy][cx];
				else
					outDelta[blockIdx.z][cy * poolSize + h][cx * poolSize + w] = 0;
			}
		}
	}
}

__global__ void _CNN_D_ReLU_Kernel(GMat3D inDelta, float ***output, int stride, float ***outDelta){
	int cx = blockDim.x * blockIdx.x + threadIdx.x;
	int cy = blockDim.y * blockIdx.y + threadIdx.y;

	if(cx < inDelta.d1 && cy < inDelta.d2){
		if(output[blockIdx.z][cy][cx] > 0){
			outDelta[blockIdx.z][cy * stride][cx * stride] = inDelta.data[blockIdx.z][cy][cx];
		}
		else{
			outDelta[blockIdx.z][cy * stride][cx * stride] = 0.f;
		}
	}
}

__global__ void _CNN_Correlation_Kernel(GMat4D inDelta, GMat4D kernel, int padding, GMat4D outDelta){
	const int cx = blockDim.x * blockIdx.x + threadIdx.x; // outDelta_x
	const int cy = blockDim.y * blockIdx.y + threadIdx.y; // outDelta_y
	const int x_in = 31 + kernel.d1;
	const int y_in = 31 + kernel.d2;
	const int im = blockIdx.z % outDelta.d3;
	const int nSample = blockIdx.z / outDelta.d3;

	__shared__ float _inDelta[42][42];
	float _sum = 0.f;
	// (in - k) / st + 1 = out
	// (out - 1) * st + k = in
	// (32 - 1) + 11 = 42

	for(int om = 0; om < inDelta.d3; ++om){
		for(int y = threadIdx.y, _y = cy - padding; y < y_in && _y < inDelta.d2 + padding; y += 32, _y += 32){
			for(int x = threadIdx.x, _x = cx - padding; x < x_in && _x < inDelta.d1 + padding; x += 32, _x += 32){
				if(_x >= 0 && _x < inDelta.d1 && _y >= 0 && _y < inDelta.d2)
					_inDelta[y][x] = inDelta.data[nSample][om][_y][_x];
				else
					_inDelta[y][x] = 0.f;
			}
		}
		__syncthreads();

		if(cy < outDelta.d2 && cx < outDelta.d1){
			for(int h = 0, _h = kernel.d2 - 1; h < kernel.d2; ++h, --_h){
				for(int w = 0, _w = kernel.d1 - 1; w < kernel.d1; ++w, --_w){
					_sum += _inDelta[threadIdx.y + h][threadIdx.x + w] * kernel.data[im][om][_h][_w];
				}
			}
		}
		__syncthreads();
	}
	if(cy < outDelta.d2 && cx < outDelta.d1) outDelta.data[nSample][im][cy][cx] = _sum;
}

__global__ void _CNN_ModifyKernel_Kernel(GMat4D delta, GMat4D kernel, GMat4D input, int padding, float ****kMoment, float learnRate, float momentRate){
	__shared__ float sumArr[32][32];
	float _sum = 0.f;

	const int im = blockIdx.z / kernel.d3;
	const int om = blockIdx.z % kernel.d3;


	for(int n = 0; n < delta.d4; ++n){
		for(int y = 0; y < delta.d2; y += 32){
			int in_y = y + threadIdx.y + blockIdx.y - padding;
			if(in_y >= 0 && in_y < input.d2 && y + threadIdx.y < delta.d2){
				for(int x = 0; x < delta.d1; x += 32){
					int in_x = x + threadIdx.x + blockIdx.x - padding;
					if(in_x >= 0 && in_x < input.d1 && x + threadIdx.x < delta.d1)
						_sum += delta.data[n][om][threadIdx.y + y][threadIdx.x + x] * input.data[n][im][in_y][in_x];
				}
			}
		}
	}
	
	sumArr[threadIdx.y][threadIdx.x] = _sum;
	__syncthreads();

	if(threadIdx.y < 16) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 16][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 8) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 8][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 4) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 4][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 2) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 2][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 1) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 1][threadIdx.x]; __syncthreads();

	if(threadIdx.x < 16 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 16]; __syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 8]; __syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 4]; __syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 2]; __syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 1]; __syncthreads();
	
	if(threadIdx.x == 0 && threadIdx.y == 0){
		float val = sumArr[0][0] / input.d4;
		float G = kMoment[im][om][blockIdx.y][blockIdx.x] = momentRate * kMoment[im][om][blockIdx.y][blockIdx.x] + (1 - momentRate) * val * val;
		kernel.data[im][om][blockIdx.y][blockIdx.x] += learnRate / sqrt(G + 0.000001f) * val;
	}		
}

__global__ void _CNN_ModifyBias_Kernel(GMat4D delta, float *bias, float *bMoment, float learnRate, float momentRate){
	__shared__ float sumArr[32][32];
	float _sum = 0.f;

	for(int n = 0; n < delta.d4; ++n){
		for(int y = 0; y < delta.d2; y += 32){
			for(int x = 0; x < delta.d1; x += 32){
				if(y + threadIdx.y < delta.d2 && x + threadIdx.x < delta.d1)
					_sum += delta.data[n][blockIdx.x][y + threadIdx.y][x + threadIdx.x];
			}
		}
	}

	sumArr[threadIdx.y][threadIdx.x] = _sum;
	__syncthreads();

	if(threadIdx.y < 16) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 16][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 8) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 8][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 4) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 4][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 2) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 2][threadIdx.x]; __syncthreads();
	if(threadIdx.y < 1) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y + 1][threadIdx.x]; __syncthreads();

	if(threadIdx.x < 16 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 16]; __syncthreads();
	if(threadIdx.x < 8 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 8]; __syncthreads();
	if(threadIdx.x < 4 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 4]; __syncthreads();
	if(threadIdx.x < 2 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 2]; __syncthreads();
	if(threadIdx.x < 1 && threadIdx.y == 0) sumArr[threadIdx.y][threadIdx.x] += sumArr[threadIdx.y][threadIdx.x + 1]; __syncthreads();

	if(threadIdx.x == 0 && threadIdx.y == 0){
		float val = sumArr[0][0] / delta.d4;		
		float G = momentRate * bMoment[blockIdx.x] + (1 - momentRate) * val * val;
		bias[blockIdx.x] += learnRate / sqrt(G + 0.000001f) * val;
	}
}
