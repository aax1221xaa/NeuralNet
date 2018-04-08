#include "ANN_Algorithm.h"
#include "_ANN_Algorithm.cuh"

/******************************************************

						ANN_Input

*******************************************************/

ANN_Input::~ANN_Input(){
	ReleaseGMat(output);
}

ANN_Input::ANN_Input(Dim inputSize){
	name = "ANN_Input";
	output = CreateGMat(inputSize);
}

GMatrix ANN_Input::GetOutput(){
	return output;
}

int ANN_Input::GetBatchSize(){
	return output.d2;
}

void ANN_Input::ConvertInput(Matrix &sample, int _start){
	ErrorChk(cudaMemcpy(output.gmem1d, sample.mem2d[_start], sizeof(float) * output.t_elements, cudaMemcpyHostToDevice));
}



/******************************************************

						CNN_To_ANN

*******************************************************/

CNN_To_ANN::~CNN_To_ANN(){
	ReleaseGMat(output);
}

CNN_To_ANN::CNN_To_ANN(GMatrix &_input){
	name = "CNN_To_ANN";
	attr = 0;

	input = _input;
	output = CreateGMat(Dim(_input.d1 * _input.d2 * _input.d3, _input.d4), _input.gmem1d);
}

GMatrix CNN_To_ANN::GetOutput(){
	return output;
}

GMatrix CNN_To_ANN::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	ANN_To_CNN *atc = new ANN_To_CNN(delta, this);
	layer.push_back(atc);

	return atc->GetOutput();
}



/******************************************************

					 ANN_ProductMat

*******************************************************/

ANN_ProductMat::~ANN_ProductMat(){
	ReleaseGMat(weight);
	ReleaseGMat(bias);
	ReleaseGMat(output);
}

ANN_ProductMat::ANN_ProductMat(GMatrix &_input, int outputSize){
	name = "ANN_ProductMat";
	attr = -1;

	input = _input;
	weight = CreateGMat(Dim(outputSize, _input.d1));
	bias = CreateGMat(Dim(outputSize));
	output = CreateGMat(Dim(outputSize, _input.d2));

	ZeroGMat(bias);

	float *randVal = new float[weight.t_elements];
	for(size_t i = 0; i < weight.t_elements; ++i) randVal[i] = random(-0.2f, 0.2f);

	ErrorChk(cudaMemcpy(weight.gmem1d, randVal, weight.t_elements * sizeof(float), cudaMemcpyHostToDevice));
	delete[] randVal;
}

void ANN_ProductMat::Run(){
	ErrorChk(_ANN_ProductMat(input, weight, bias, output));
}

GMatrix ANN_ProductMat::GetOutput(){
	return output;
}

GMatrix ANN_ProductMat::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	GMatrix outDelta;

	if(attr){
		ANN_D_ProductMat *dpro = new ANN_D_ProductMat(delta, this);
		layer.push_back(dpro);
		outDelta = dpro->GetOutput();
	}
	return outDelta;
}

void ANN_ProductMat::SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate){
	ANN_Modify *modify = new ANN_Modify(delta, this, learnRate, momentRate);
	layer.push_back(modify);
}




/******************************************************

						ANN_ReLU

*******************************************************/

ANN_ReLU::~ANN_ReLU(){

}

ANN_ReLU::ANN_ReLU(GMatrix &_io){
	name = "ANN_ReLU";
	attr = 0;

	io = _io;
}

void ANN_ReLU::Run(){
	ErrorChk(_ANN_ReLU(io));
}

GMatrix ANN_ReLU::GetOutput(){
	return io;
}

GMatrix ANN_ReLU::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	ANN_D_ReLU *drelu = new ANN_D_ReLU(delta, this);
	layer.push_back(drelu);

	return drelu->GetOutput();
}




/******************************************************

						SoftMax

*******************************************************/

SoftMax::~SoftMax(){

}

SoftMax::SoftMax(GMatrix &_io){
	name = "SoftMax";
	attr = 0;

	io = _io;
}

void SoftMax::Run(){
	ErrorChk(_SoftMax(io));
}

GMatrix SoftMax::GetOutput(){
	return io;
}




/******************************************************

					  Cross Entropy

*******************************************************/

CrossEntropy::~CrossEntropy(){
	ReleaseGMat(target);
	ReleaseGMat(delta);
	cudaFree(cost);
}

CrossEntropy::CrossEntropy(GMatrix &_output){
	name = "CrossEntropy";

	output = _output;
	target = CloneGMat(_output);
	delta = CloneGMat(_output);
	cudaMalloc(&cost, sizeof(float));
}

GMatrix CrossEntropy::GetDelta(){
	return delta;
}

float CrossEntropy::CalcCost(Matrix &_target, int _start){
	ErrorChk(cudaMemcpy(target.gmem1d, _target.mem2d[_start], sizeof(float) * target.t_elements, cudaMemcpyHostToDevice));
	ErrorChk(_CrossEntropy(target, output, cost));
	ErrorChk(_CrossEntropyDelta(target, output, delta));

	float _cost = 0.f;

	ErrorChk(cudaMemcpy(&_cost, cost, sizeof(float), cudaMemcpyDeviceToHost));
	return _cost;
}



/******************************************************

					   ANN_D_ReLU

*******************************************************/

ANN_D_ReLU::~ANN_D_ReLU(){

}

ANN_D_ReLU::ANN_D_ReLU(GMatrix &_ioDelta, ANN_ReLU *p){
	name = "ANN_D_ReLU";
	attr = 0;

	ioDelta = _ioDelta;
	output = p->io;
}

void ANN_D_ReLU::Run(){
	ErrorChk(_ANN_D_ReLU(ioDelta, output));
}

GMatrix ANN_D_ReLU::GetOutput(){
	return ioDelta;
}



/******************************************************

					ANN_D_ProductMat

*******************************************************/

ANN_D_ProductMat::~ANN_D_ProductMat(){
	ReleaseGMat(outDelta);
}

ANN_D_ProductMat::ANN_D_ProductMat(GMatrix &_inDelta, ANN_ProductMat *p){
	name = "ANN_D_ProductMat";
	attr = 0;

	inDelta = _inDelta;
	weight = p->weight;
	outDelta = CloneGMat(p->input);
}

void ANN_D_ProductMat::Run(){
	ErrorChk(_ANN_D_ProductMat(inDelta, weight, outDelta));
}

GMatrix ANN_D_ProductMat::GetOutput(){
	return outDelta;
}



/******************************************************

					 ANN_To_CNN

*******************************************************/

ANN_To_CNN::~ANN_To_CNN(){
	ReleaseGMat(outDelta);
}

ANN_To_CNN::ANN_To_CNN(GMatrix &_inDelta, CNN_To_ANN *p){
	name = "ANN_To_CNN";
	attr = 0;

	outDelta = CreateGMat(Dim(p->input.d1, p->input.d2, p->input.d3, p->input.d4), _inDelta.gmem1d);
}

GMatrix ANN_To_CNN::GetOutput(){
	return outDelta;
}



/******************************************************

					   ANN_Modify

*******************************************************/

ANN_Modify::~ANN_Modify(){
	ReleaseGMat(wMoment);
	ReleaseGMat(bMoment);
}

ANN_Modify::ANN_Modify(GMatrix &_delta, ANN_ProductMat *p, float _learnRate, float _momentRate){
	name = "ANN_Modify";
	attr = 0;
	
	delta = _delta;
	weight = p->weight;
	bias = p->bias;
	input = p->input;

	wMoment = CloneGMat(p->weight);
	bMoment = CloneGMat(p->bias);

	ZeroGMat(wMoment);
	ZeroGMat(bMoment);

	learnRate = _learnRate;
	momentRate = _momentRate;
}

void ANN_Modify::Run(){
	ErrorChk(_ANN_ModifyWeight(delta, weight, input, wMoment, learnRate, momentRate));
	ErrorChk(_ANN_ModifyBias(delta, bias, bMoment, learnRate, momentRate));
}