#include "NN_Base.h"


void TranceMat(Mat &dst, Matrix &src){
	for(int n = 0; n < src.d4; ++n){
		for(int c = 0; c < src.d3; ++c){
			for(int y = 0; y < src.d2; ++y){
				float *fpt = dst.ptr<float>(y);
				for(int x = 0; x < src.d1; ++x){
					fpt[x * src.d3 + c + (n * src.d1)] = src.mem4d[n][c][y][x];
				}
			}
		}
	}
}

void TranceMat2(Mat &dst, Matrix &src){
	for(int n = 0; n < src.d4; ++n){
		for(int c = 0; c < src.d3; ++c){
			for(int y = 0; y < src.d2; ++y){
				float *fpt = dst.ptr<float>(c * src.d2 + y);
				for(int x = 0; x < src.d1; ++x){
					fpt[n * src.d1 + x] = src.mem4d[n][c][y][x];
				}
			}
		}
	}
}

/******************************************************

						NN_Layer

*******************************************************/

NN_Layer::~NN_Layer(){

}

void NN_Layer::Run(){

}

#ifdef DEBUG
Mat NN_Layer::GetMat(){
	return monitor;
}
#endif

GMatrix NN_Layer::GetOutput(){
	return GMatrix();
}

GMatrix NN_Layer::SetBackward(vector<NN_Layer*> &layer, GMatrix &delta){
	return delta;
}

void NN_Layer::SetModify(vector<NN_Layer*> &layer, GMatrix &delta, float learnRate, float momentRate){

}




/******************************************************

					   StartLayer

*******************************************************/

StartLayer::~StartLayer(){

}

GMatrix StartLayer::GetOutput(){
	return GMatrix();
}

int StartLayer::GetBatchSize(){
	return 0;
}

void StartLayer::ConvertInput(Matrix &sample, int _start){

}

#ifdef DEBUG
Mat StartLayer::GetMat(){
	return Mat();
}
#endif



/******************************************************

						EndLayer

*******************************************************/

EndLayer::~EndLayer(){

}

GMatrix EndLayer::GetDelta(){
	return GMatrix();
}

float EndLayer::CalcCost(Matrix & target, int _start){
	return 0.f;
}