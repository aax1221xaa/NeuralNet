#include <iostream>
#include <vld.h>
#include "NN.h"
//#include "MNIST_Data.h"
#include "CIFAR_10.h"

using namespace std;

#define TEST
//#define TRAIN

#if defined(TRAIN)

int main(){
	srand((uint)time(NULL));

	
	CIFAR_10 data("E:\\CppWork\\OpencvEx\\image\\cifar-10-batches-bin, 1");

	NN test;
	GMatrix out;

	out = test.StartSet(new CNN_Input(Dim(32, 32, 3, 100)));
	out = test.push_back(new CNN_Convolution(out, Dim(5, 5), 1, 0, Dim(28, 28, 8)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_Convolution(out, Dim(5, 5), 1, 0, Dim(24, 24, 16)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_MaxPool(out, 2));	// 12, 12, 16
	out = test.push_back(new CNN_Convolution(out, Dim(3, 3), 1, 0, Dim(10, 10, 24)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_Convolution(out, Dim(3, 3), 1, 0, Dim(8, 8, 32)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_MaxPool(out, 2));	// 4 4 32
	out = test.push_back(new CNN_To_ANN(out));
	out = test.push_back(new ANN_ProductMat(out, 128));
	out = test.push_back(new ANN_ReLU(out));
	out = test.push_back(new ANN_ProductMat(out, 10));
	out = test.push_back(new SoftMax(out));
	test.EndSet(new CrossEntropy(out), 0.0001f, 0.9f);

	test.Training(data.sample, data.label, 0, 10000, 10, 0.1f);
	test.Save("E:\\CppWork\\OpencvEx\\ann_data\\test2.wb");
	
	return 0;
}

#endif

#if defined(TEST)

int main(){
	srand((uint)time(NULL));

	
	CIFAR_10 data("E:\\CppWork\\OpencvEx\\image\\cifar-10-batches-bin");

	NN test;
	GMatrix out;

	out = test.StartSet(new CNN_Input(Dim(32, 32, 3, 1)));
	out = test.push_back(new CNN_Convolution(out, Dim(5, 5), 1, 0, Dim(28, 28, 8)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_Convolution(out, Dim(5, 5), 1, 0, Dim(24, 24, 16)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_MaxPool(out, 2));	// 12, 12, 16
	out = test.push_back(new CNN_Convolution(out, Dim(3, 3), 1, 0, Dim(10, 10, 24)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_Convolution(out, Dim(3, 3), 1, 0, Dim(8, 8, 32)));
	out = test.push_back(new CNN_ReLU(out, 1));
	out = test.push_back(new CNN_MaxPool(out, 2));	// 4 4 32
	out = test.push_back(new CNN_To_ANN(out));
	out = test.push_back(new ANN_ProductMat(out, 128));
	out = test.push_back(new ANN_ReLU(out));
	out = test.push_back(new ANN_ProductMat(out, 10));
	out = test.push_back(new SoftMax(out));
	test.EndSet(out);

	test.Load("E:\\CppWork\\OpencvEx\\ann_data\\test2.wb");
	
	Matrix result = CreateMat(Dim(10, 1));

	int cnt = 0;
	for(int i = 0; i < 10000;++i){
		test.Predict(data.sample, i, result);
		float val = -1.f;
		int idx = 0;
		
		for(int k = 0; k < 10; ++k){
			if(result.mem1d[k] > val){
				val = result.mem1d[k];
				idx = k;
			}
		}
		if(data.label.mem2d[i][idx] == 1.f) ++cnt;
	}
	cout << cnt << endl;
	ReleaseMat(result);
	
	return 0;
}

#endif