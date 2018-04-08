#ifndef NN_H
#define NN_H

#include <iostream>
#include <string>
#include <vector>

#include "NN_Misc.h"
#include "NN_Base.h"
#include "ANN_Algorithm.h"
#include "CNN_Algorithm.h"


using namespace std;



/******************************************************

						  NN

*******************************************************/

class NN{
public:
	StartLayer *cvtInput;
	vector<NN_Layer*> _forward;
	vector<NN_Layer*> _backward;
	vector<NN_Layer*> _modify;
	EndLayer *costFunc;

	int nWeight;
	int batchSize;

	GMatrix output;

	NN();
	~NN();
	GMatrix StartSet(StartLayer *layer);
	GMatrix push_back(NN_Layer *layer);
	void EndSet(EndLayer *layer, float learnRate, float momentRate);
	void EndSet(GMatrix &_output);
	void Training(Matrix &sample, Matrix &target, int _start, int _end, int iter, float limit);
	void Predict(Matrix &input, int _start, Matrix &_output);

	void Save(const char path[]);
	void Load(const char path[]);
};

#endif