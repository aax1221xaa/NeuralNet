#include "CNN_Algorithm.h"
#include "_CNN_Algorithm.cuh"


/******************************************************

						CNN_Input

*******************************************************/

CNN_Input::~CNN_Input(){
	ReleaseGMat(output);
}

CNN_Input::CNN_Input(Dim inputSize){
	name = "CNN_Input";

	output = CreateGMat(inputSize);
}


GMatrix CNN_Input::GetOutput(){
	return output;
}

int CNN_Input::GetBatchSize(){
	return output.d4;
}

void CNN_Input::ConvertInput(Matrix &sample, int _start){
	ErrorChk(cudaMemcpy(output.gmem1d, sample.mem4d[_start][0][0], output.t_elements * sizeof(float), cudaMemcpyHostToDevice));
}



/******************************************************

					CNN_Convolution

*******************************************************/

CNN_Convolution::~CNN_Convolution(){
	ReleaseGMat(kernel);
	ReleaseGMat(bias);
	ReleaseGMat(output);
}

CNN_Convolution::CNN_Convolution(GMatrix &_input, Dim kernelSize, int _stride, int _padding, Dim outputSize){
	name = "CNN_Convolution";
	attr = -1;

	input = _input;
	kernel = CreateGMat(Dim(kernelSize.d1, kernelSize.d2, outputSize.d3, _input.d3));
	bias = CreateGMat(Dim(outputSize.d3));
	output = CreateGMat(Dim(outputSize.d1, outputSize.d2, outputSize.d3, _input.d4));

	stride = _stride;
	padding = _padding;

	float *randVal = new float[kernel.t_elements];
	
	for(size_t i = 0; i < kernel.t_elements; ++i) randVal[i] = random(-0.2f, 0.2f);
	ErrorChk(cudaMemcpy(kernel.gmem1d, randVal, sizeof(float) * kernel.t_elements, cudaMemcpyHostToDevice));
	ZeroGMat(bias);

	delete[] randVal;
}


void CNN_Convolution::Run(){
	ErrorChk(_CNN_Convolution(input, stride, padding, kernel, bias, output));
}

GMatrix CNN_Convolution::GetOutput(){
	return output;
}

GMatrix CNN_Convolution::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	GMatrix outDelta;

	if(attr){
		CNN_Correlation *corr = new CNN_Correlation(delta, this);
		layer.push_back(corr);
		outDelta = corr->GetOutput();
	}

	return outDelta;
}

void CNN_Convolution::SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate){
	CNN_Modify *modify = new CNN_Modify(delta, this, learnRate, momentRate);
	layer.push_back(modify);
}




/******************************************************

						CNN_ReLU

*******************************************************/


CNN_ReLU::~CNN_ReLU(){

}

CNN_ReLU::CNN_ReLU(GMatrix &_io, int _stride){
	name = "CNN_ReLU";
	attr = 0;

	io = _io;
	stride = _stride;
}


void CNN_ReLU::Run(){
	ErrorChk(_CNN_ReLU(io));
}

GMatrix CNN_ReLU::GetOutput(){
	return io;
}

GMatrix CNN_ReLU::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	CNN_D_ReLU *drelu = new CNN_D_ReLU(delta, this);
	layer.push_back(drelu);

	return drelu->GetOutput();
}



/******************************************************

					   CNN_MaxPool

*******************************************************/

CNN_MaxPool::~CNN_MaxPool(){
	ReleaseGMark(mark);
	ReleaseGMat(output);
}

CNN_MaxPool::CNN_MaxPool(GMatrix &_input, int _poolSize){
	name = "CNN_MaxPool";
	attr = 0;

	poolSize = _poolSize;
	input = _input;
	mark = CreateGMark(Dim(_input.d1 / _poolSize, _input.d2 / _poolSize, _input.d3 * _input.d4));
	output = CreateGMat(Dim(_input.d1 / _poolSize, _input.d2 / _poolSize, _input.d3, _input.d4));
}


void CNN_MaxPool::Run(){
	ErrorChk(_CNN_MaxPool(input, mark, poolSize, output));
}

GMatrix CNN_MaxPool::GetOutput(){
	return output;
}

GMatrix CNN_MaxPool::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	CNN_UnPool *unpool = new CNN_UnPool(delta, this);
	layer.push_back(unpool);

	return unpool->GetOutput();
}




/******************************************************

					   CNN_UnPool

*******************************************************/

CNN_UnPool::~CNN_UnPool(){
	ReleaseGMat(outDelta);
}

CNN_UnPool::CNN_UnPool(GMatrix &_inDelta, CNN_MaxPool *p){
	name = "CNN_UnPool";
	attr = 0;

	inDelta = _inDelta;
	mark = p->mark;
	poolSize = p->poolSize;
	outDelta = CreateGMat(Dim(p->input.d1, p->input.d2, p->input.d3, p->input.d4));
}


void CNN_UnPool::Run(){
	ErrorChk(_CNN_UnPool(inDelta, mark, poolSize, outDelta));
}

GMatrix CNN_UnPool::GetOutput(){
	return outDelta;
}




/******************************************************

					   CNN_D_ReLU

*******************************************************/

CNN_D_ReLU::~CNN_D_ReLU(){
	ReleaseGMat(outDelta);
}

CNN_D_ReLU::CNN_D_ReLU(GMatrix &_inDelta, CNN_ReLU *p){
	name = "CNN_D_ReLU";
	attr = 0;

	inDelta = _inDelta;
	output = p->io;
	stride = p->stride;

	outDelta = CreateGMat(Dim(
		(_inDelta.d1 - 1) * stride + 1,
		(_inDelta.d2 - 1) * stride + 1,
		_inDelta.d3, _inDelta.d4));
	
	ZeroGMat(outDelta);
}

	
void CNN_D_ReLU::Run(){
	ErrorChk(_CNN_D_ReLU(inDelta, output, stride, outDelta));
}

GMatrix CNN_D_ReLU::GetOutput(){
	return outDelta;
}




/******************************************************

					 CNN_Correlation

*******************************************************/

CNN_Correlation::~CNN_Correlation(){
	ReleaseGMat(outDelta);
}

CNN_Correlation::CNN_Correlation(GMatrix &_inDelta, CNN_Convolution *p){
	name = "CNN_Correlation";
	attr = 0;

	padding = p->kernel.d1 - 1;

	inDelta = _inDelta;
	kernel = p->kernel;
	outDelta = CreateGMat(Dim(p->input.d1, p->input.d2, p->input.d3, p->input.d4));
}


void CNN_Correlation::Run(){
	ErrorChk(_CNN_Correlation(inDelta, kernel, padding, outDelta));
}

GMatrix CNN_Correlation::GetOutput(){
	return outDelta;
}




/******************************************************

					   CNN_Modify

*******************************************************/

CNN_Modify::~CNN_Modify(){
	ReleaseGMat(kMoment);
	ReleaseGMat(bMoment);
}

CNN_Modify::CNN_Modify(GMatrix &_delta, CNN_Convolution *p, float _learnRate, float _momentRate){
	name = "CNN_Modify";
	attr = 0;

	padding = p->padding;

	delta = _delta;
	kernel = p->kernel;
	bias = p->bias;
	input = p->input;

	kMoment = CloneGMat(p->kernel);
	bMoment = CloneGMat(p->bias);
	learnRate = _learnRate;
	momentRate = _momentRate;

	ZeroGMat(kMoment);
	ZeroGMat(bMoment);
}

void CNN_Modify::Run(){
	ErrorChk(_CNN_ModifyKernel(delta, kernel, input, padding, kMoment, learnRate, momentRate));
	ErrorChk(_CNN_ModifyBias(delta, bias, bMoment, learnRate, momentRate));
}
