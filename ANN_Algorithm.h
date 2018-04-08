#ifndef ANN_ALGORITHM_H
#define ANN_ALGORITHM_H

#include "NN_Misc.h"
#include "NN_Base.h"



/******************************************************

						ANN_Input

*******************************************************/

class ANN_Input : public StartLayer{
public:
	GMatrix output;

	~ANN_Input();
	ANN_Input(Dim inputSize);
	GMatrix GetOutput();
	int GetBatchSize();
	void ConvertInput(Matrix &sample, int _start);
};



/******************************************************

						CNN_To_ANN

*******************************************************/

class CNN_To_ANN : public NN_Layer{
public:
	GMatrix input;
	GMatrix output;

	~CNN_To_ANN();
	CNN_To_ANN(GMatrix &_input);

	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
};



/******************************************************

					 ANN_ProductMat

*******************************************************/

class ANN_ProductMat : public NN_Layer{
public:
	GMatrix input;
	GMatrix weight;
	GMatrix bias;
	GMatrix output;

	~ANN_ProductMat();
	ANN_ProductMat(GMatrix &_input, int outputSize);

	void Run();
	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
	void SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate);
};



/******************************************************

						ANN_ReLU

*******************************************************/

class ANN_ReLU : public NN_Layer{
public:
	GMatrix io;

	~ANN_ReLU();
	ANN_ReLU(GMatrix &_io);

	void Run();
	GMatrix GetOutput();
	GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
};



/******************************************************

						SoftMax

*******************************************************/

class SoftMax : public NN_Layer{
public:
	GMatrix io;

	~SoftMax();
	SoftMax(GMatrix &_io);

	void Run();
	GMatrix GetOutput();
};



/******************************************************

					  Cross Entropy

*******************************************************/

class CrossEntropy : public EndLayer{
public:
	GMatrix output;
	GMatrix target;
	GMatrix delta;
	float *cost;

	~CrossEntropy();
	CrossEntropy(GMatrix &_output);
	GMatrix GetDelta();
	float CalcCost(Matrix &_target, int _start);
};



/******************************************************

					   ANN_D_ReLU

*******************************************************/

class ANN_D_ReLU : public NN_Layer{
public:
	GMatrix ioDelta;
	GMatrix output;

	~ANN_D_ReLU();
	ANN_D_ReLU(GMatrix &_ioDelta, ANN_ReLU *p);

	void Run();
	GMatrix GetOutput();
};



/******************************************************

					ANN_D_ProductMat

*******************************************************/

class ANN_D_ProductMat : public NN_Layer{
public:
	GMatrix inDelta;
	GMatrix weight;
	GMatrix outDelta;

	~ANN_D_ProductMat();
	ANN_D_ProductMat(GMatrix &_inDelta, ANN_ProductMat *p);

	void Run();
	GMatrix GetOutput();
};



/******************************************************

					 ANN_To_CNN

*******************************************************/

class ANN_To_CNN : public NN_Layer{
public:
	GMatrix outDelta;

	~ANN_To_CNN();
	ANN_To_CNN(GMatrix &_inDelta, CNN_To_ANN *p);

	GMatrix GetOutput();
};



/******************************************************

					   ANN_Modify

*******************************************************/

class ANN_Modify : public NN_Layer{
public:
	GMatrix delta;
	GMatrix weight;
	GMatrix bias;
	GMatrix input;

	GMatrix wMoment;
	GMatrix bMoment;

	float learnRate;
	float momentRate;

	~ANN_Modify();
	ANN_Modify(GMatrix &_delta, ANN_ProductMat *p, float _learnRate, float _momentRate);

	void Run();
};

#endif