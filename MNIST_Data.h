#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <opencv2\opencv.hpp>
#include "NN_Misc.h"

class MNIST_Data{
	void Create(const char *imgPath, const char *labelPath, int _nSample);

public:
	Matrix sample;
	Matrix result;

	MNIST_Data();
	MNIST_Data(const char *path, const char *labelPath, int _nSample);
	~MNIST_Data();

	void Set(const char *path, const char *labelPath, int _nSample);
	void Release();
};

#endif