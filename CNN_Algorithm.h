#ifndef CNN_ALGORITHM_H
#define CNN_ALGORITHM_H

#include "NN_Misc.h"
#include "NN_Base.h"



/******************************************************

						CNN_Input

*******************************************************/

class CNN_Input : public StartLayer{
public:
	GMatrix output;

	~CNN_Input();
	CNN_Input(Dim inputSize);

	GMatrix GetOutput();
	int GetBatchSize();
	void ConvertInput(Matrix &sample, int _start);

};



/******************************************************

					CNN_Convolution

*******************************************************/

class CNN_Convolution : public NN_Layer{
public:
	GMatrix input;
	GMatrix kernel;
	GMatrix bias;
	GMatrix output;

	int stride;
	int padding;

	~CNN_Convolution();
	CNN_Convolution(GMatrix &_input, Dim kernelSize, int _stride, int _padding, Dim outputSize);

	void Run();
	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
	void SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate);
};



/******************************************************

						CNN_ReLU

*******************************************************/

class CNN_ReLU : public NN_Layer{
public:
	GMatrix io;
	int stride;

	~CNN_ReLU();
	CNN_ReLU(GMatrix &_io, int _stride);

	void Run();
	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
};



/******************************************************

					   CNN_MaxPool

*******************************************************/

class CNN_MaxPool : public NN_Layer{
public:
	GMatrix input;
	GMark mark;
	GMatrix output;

	int poolSize;

	~CNN_MaxPool();
	CNN_MaxPool(GMatrix &_input, int _poolSize);

	void Run();
	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
};




/******************************************************

					   CNN_UnPool

*******************************************************/

class CNN_UnPool : public NN_Layer{
public:
	GMatrix inDelta;
	GMark mark;
	GMatrix outDelta;

	int poolSize;

	~CNN_UnPool();
	CNN_UnPool(GMatrix &_inDelta, CNN_MaxPool *p);

	void Run();
	GMatrix GetOutput();
};



/******************************************************

					   CNN_D_ReLU

*******************************************************/

class CNN_D_ReLU : public NN_Layer{
public:
	GMatrix inDelta;
	GMatrix output;
	GMatrix outDelta;

	int stride;

	~CNN_D_ReLU();
	CNN_D_ReLU(GMatrix &_inDelta, CNN_ReLU *p);
	
	void Run();
	GMatrix GetOutput();
};



/******************************************************

					 CNN_Correlation

*******************************************************/

class CNN_Correlation : public NN_Layer{
public:
	GMatrix inDelta;
	GMatrix kernel;
	GMatrix outDelta;

	int padding;

	~CNN_Correlation();
	CNN_Correlation(GMatrix &_inDelta, CNN_Convolution *p);

	void Run();
	GMatrix GetOutput();
};



/******************************************************

					   CNN_Modify

*******************************************************/

class CNN_Modify : public NN_Layer{
public:
	GMatrix delta;
	GMatrix kernel;
	GMatrix bias;
	GMatrix input;

	GMatrix kMoment;
	GMatrix bMoment;

	float learnRate;
	float momentRate;
	int padding;

	~CNN_Modify();
	CNN_Modify(GMatrix &_delta, CNN_Convolution *p, float _learnRate, float _momentRate);
	void Run();
};

#endif