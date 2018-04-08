#ifndef _NN_BASE_H
#define _NN_BASE_H

#include "NN_Misc.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>


using namespace std;
using namespace cv;


//#define DEBUG


void TranceMat(Mat &dst, Matrix &src);
void TranceMat2(Mat &dst, Matrix &src);


/******************************************************

						NN_Layer

*******************************************************/

class NN_Layer{
public:
	string name;
	int attr;
	
	virtual ~NN_Layer();
	virtual void Run();

#ifdef DEBUG
	Mat monitor;
	virtual Mat GetMat();
#endif

	virtual GMatrix GetOutput();
	virtual GMatrix SetBackward(vector<NN_Layer*> &layer, GMatrix &delta);
	virtual void SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate);
};


/******************************************************

					   StartLayer

*******************************************************/

class StartLayer{
public:
	string name;

	virtual ~StartLayer();
	virtual GMatrix GetOutput();
	virtual int GetBatchSize();
	virtual void ConvertInput(Matrix &sample, int _start);

#ifdef DEBUG
	Mat monitor;
	virtual Mat GetMat();
#endif
};


/******************************************************

						EndLayer

*******************************************************/

class EndLayer{
public:
	string name;

	virtual ~EndLayer();
	virtual GMatrix GetDelta();
	virtual float CalcCost(Matrix &target, int _start);
};

#endif