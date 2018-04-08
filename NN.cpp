#include "NN.h"



/******************************************************

						  NN

*******************************************************/


NN::NN(){
	cvtInput = NULL;
	nWeight = 0;
	batchSize = 0;
	costFunc = NULL;
}

NN::~NN(){
	delete cvtInput;

	for(NN_Layer *p : _forward) delete p;
	for(NN_Layer *p : _backward) delete p;
	for(NN_Layer *p : _modify) delete p;

	delete costFunc;
	ReleaseGMat(output);
}

GMatrix NN::StartSet(StartLayer *layer){
	cvtInput = layer;
	batchSize = layer->GetBatchSize();

	return layer->GetOutput();
}

GMatrix NN::push_back(NN_Layer *layer){
	GMatrix out;

	if(layer->attr){
		layer->attr = nWeight;
		++nWeight;
	}

	_forward.push_back(layer);
	out = layer->GetOutput();

	return out;
}

void NN::EndSet(EndLayer *layer, float learnRate, float momentRate){
	costFunc = layer;
	GMatrix outDelta = layer->GetDelta();

	for(vector<NN_Layer*>::reverse_iterator iter = _forward.rbegin(); iter != _forward.rend(); ++iter){
		(*iter)->SetModify(_modify, outDelta, learnRate, momentRate);
		outDelta = (*iter)->SetBackward(_backward, outDelta);
	}
}

void NN::EndSet(GMatrix &_output){
	output = _output;
}

void NN::Training(Matrix &sample, Matrix &target, int _start, int _end, int iter, float limit){
	for(int s = 0; s < iter; ++s){
		for(int i = _start; i < _end; i += batchSize){
			cvtInput->ConvertInput(sample, i);
			for(NN_Layer *p : _forward) p->Run();
			float cost = costFunc->CalcCost(target, i);
			
			printf("#%d (%d ~ %d) Loss = %f\n", s, i, i + batchSize, cost);

			if(cost > limit){
				for(NN_Layer *p : _backward) p->Run();
				for(NN_Layer *p : _modify) p->Run();
			}
		}
	}
}

void NN::Predict(Matrix &input, int _start, Matrix &_output){
	cvtInput->ConvertInput(input, _start);
	for(NN_Layer *p : _forward) p->Run();

	ErrorChk(cudaMemcpy(_output.mem1d, output.gmem1d, sizeof(float) * output.t_elements, cudaMemcpyDeviceToHost)); 
}

void NN::Save(const char path[]){
	FILE *fp = NULL;

	fopen_s(&fp, path, "wb");
	if(!fp){
		cout << "경로를 열지 못하였습니다." << endl;
		return;
	}

	for(NN_Layer *p : _forward){
		if(p->name == "CNN_Convolution"){
			CNN_Convolution *conv = (CNN_Convolution*)p;
			Matrix mat = CreateMat(Dim(conv->kernel.d1, conv->kernel.d2, conv->kernel.d3, conv->kernel.d4));
			Matrix mat2 = CreateMat(Dim(conv->bias.d1));

			MemCpyD2H(conv->kernel, mat);
			MemCpyD2H(conv->bias, mat2);

			fwrite(mat.mem1d, sizeof(float), mat.t_elements, fp);
			fwrite(mat2.mem1d, sizeof(float), mat2.t_elements, fp);

			ReleaseMat(mat);
			ReleaseMat(mat2);
		}
		else if(p->name == "ANN_ProductMat"){
			ANN_ProductMat *pro = (ANN_ProductMat*)p;
			Matrix mat = CreateMat(Dim(pro->weight.d1, pro->weight.d2));
			Matrix mat2 = CreateMat(Dim(pro->bias.d1));

			MemCpyD2H(pro->weight, mat);
			MemCpyD2H(pro->bias, mat2);

			fwrite(mat.mem1d, sizeof(float), mat.t_elements, fp);
			fwrite(mat2.mem1d, sizeof(float), mat2.t_elements, fp);

			ReleaseMat(mat);
			ReleaseMat(mat2);
		}
	}

	fclose(fp);
}

void NN::Load(const char path[]){
	FILE *fp = NULL;

	fopen_s(&fp, path, "rb");
	if(!fp){
		cout << "경로를 열지 못하였습니다." << endl;
		return;
	}

	for(NN_Layer *p : _forward){
		if(p->name == "CNN_Convolution"){
			CNN_Convolution *conv = (CNN_Convolution*)p;			
			
			Matrix tmp = CreateMat(Dim(conv->kernel.d1, conv->kernel.d2, conv->kernel.d3, conv->kernel.d4));
			fread_s(tmp.mem1d, sizeof(float) * tmp.t_elements, sizeof(float), tmp.t_elements, fp); 
			MemCpyH2D(tmp, conv->kernel);
			ReleaseMat(tmp);

			Matrix tmp2 = CreateMat(Dim(conv->bias.d1));
			fread_s(tmp2.mem1d, sizeof(float) * tmp2.t_elements, sizeof(float), tmp2.t_elements, fp); 
			MemCpyH2D(tmp2, conv->bias);
			ReleaseMat(tmp2);
		}
		else if(p->name == "ANN_ProductMat"){
			ANN_ProductMat *pro = (ANN_ProductMat*)p;

			Matrix tmp = CreateMat(Dim(pro->weight.d1, pro->weight.d2));
			fread_s(tmp.mem1d, sizeof(float) * tmp.t_elements, sizeof(float), tmp.t_elements, fp); 
			MemCpyH2D(tmp, pro->weight);
			ReleaseMat(tmp);

			Matrix tmp2 = CreateMat(Dim(pro->bias.d1));
			fread_s(tmp2.mem1d, sizeof(float) * tmp2.t_elements, sizeof(float), tmp2.t_elements, fp); 
			MemCpyH2D(tmp2, pro->bias);
			ReleaseMat(tmp2);
		}
	}

	fclose(fp);
}