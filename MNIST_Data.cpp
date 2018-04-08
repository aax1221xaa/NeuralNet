#include "MNIST_Data.h"


MNIST_Data::MNIST_Data(){

}

MNIST_Data::MNIST_Data(const char *imgPath, const char *labelPath, int _nSample){
	Create(imgPath, labelPath, _nSample);
}

MNIST_Data::~MNIST_Data(){
	ReleaseMat(sample);
	ReleaseMat(result);
}

void MNIST_Data::Set(const char *imgPath, const char *labelPath, int _nSample){
	Release();
	Create(imgPath, labelPath, _nSample);
}

void MNIST_Data::Release(){
	ReleaseMat(sample);
	ReleaseMat(result);

	sample = Matrix();
	result = Matrix();
}

void MNIST_Data::Create(const char *imgPath, const char *labelPath, int _nSample){
	FILE *img_fp = NULL;
	FILE *label_fp = NULL;

	union TranceVal{
		uchar buff08[4];
		int buff32;
	}data;

	fopen_s(&img_fp, imgPath, "rb");
	if(!img_fp){
		cout << "이미지 파일을 열 수 없습니다.\n";
		return;
	}
	fopen_s(&label_fp, labelPath, "rb");
	if(!label_fp){
		cout << "라벨링 파일을 열 수 없습니다.\n";
		return;
	}

	int param[4];

	for(int i = 0; i < 4; ++i){
		for(int k = 0; k < 4; ++k){
			fread_s(&data.buff08[3 - k], sizeof(uchar), sizeof(uchar), 1, img_fp);
		}
		param[i] = data.buff32;
		cout << data.buff32 << endl;
	}

	int nSample = param[1];
	int width = param[2];
	int height = param[3];

	for(int i = 0; i < 2; ++i){
		for(int k = 0; k < 4; ++k){
			fread_s(&data.buff08[3 - k], sizeof(uchar), sizeof(uchar), 1, label_fp);
		}
		param[i] = data.buff32;
		cout << data.buff32 << endl;
	}

	if(nSample != param[1]){
		cout << "샘플 이미지와 라벨링 데이터 수가 맞지 않습니다.\n";
		fclose(img_fp);
		fclose(label_fp);
		return;
	}
	else if(nSample < _nSample){
		cout << "입력받은 샘플 수가 실제 샘플 수를 넘습니다.\n";
		fclose(img_fp);
		fclose(label_fp);
		return;
	}

	nSample = _nSample;
	sample = CreateMat(Dim(width, height, nSample));

	uchar *_tmp = new uchar[nSample];
	
	size_t size = width * height * nSample;
	uchar *_tmp2 = new uchar[size];

	result = CreateMat(Dim(10, nSample));
	SetZero(result); 

	fread_s(_tmp2, size * sizeof(uchar), sizeof(uchar), size, img_fp); 
	fread_s(_tmp, sizeof(uchar) * nSample, sizeof(uchar), nSample, label_fp); 

	fclose(img_fp);
	fclose(label_fp);

	for(size_t i = 0; i < size; ++i){
		sample.mem1d[i] = (float)_tmp2[i] / 255.f;
	}
	for(int i = 0; i < nSample; ++i){
		result.mem2d[i][_tmp[i]] = 1.f;
	}

	delete[] _tmp;
	delete[] _tmp2;
}
