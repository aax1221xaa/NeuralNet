#include "CIFAR_10.h"



CIFAR_10::~CIFAR_10(){
	ReleaseMat(sample);
	ReleaseMat(label);
}

CIFAR_10::CIFAR_10(const char dir_path[], int nFile){
	char path[100];
	char fileName[20];

	strcpy_s(path, dir_path);
	strcat_s(path, "\\batches.meta.txt");
	LoadMataFile(path);

	strcpy_s(path, dir_path);
	sprintf_s(fileName, "\\data_batch_%d.bin", nFile);
	strcat_s(path, fileName);
	LoadSampleFile(path);
}

CIFAR_10::CIFAR_10(const char dir_path[]){
	char path[100];

	strcpy_s(path, dir_path);
	strcat_s(path, "\\batches.meta.txt");
	LoadMataFile(path);

	strcpy_s(path, dir_path);
	strcat_s(path, "\\test_batch.bin");
	LoadSampleFile(path);
}

void CIFAR_10::LoadMataFile(const char dir_path[]){
	FILE *fp = NULL;

	fopen_s(&fp, dir_path, "rt");
	if(!fp){
		cout << "파일을 열지 못했습니다." << endl;
		return;
	}

	for(int i = 0; i < 10; ++i) fscanf_s(fp, "%s", name[i], sizeof(char) * 20);

	fclose(fp);
}

void CIFAR_10::LoadSampleFile(const char dir_path[]){
	FILE *fp = NULL;

	fopen_s(&fp, dir_path, "rb");
	if(!fp){
		cout << "파일을 열지 못했습니다." << endl;
		return;
	}

	sample = CreateMat(Dim(32, 32, 3, 10000));
	label = CreateMat(Dim(10, 10000));

	SetZero(label);

	size_t size = 32 * 32 * 3;
	unsigned char *tmp = new unsigned char[size];
	unsigned char nLabel = 0;

	for(int i = 0; i < 10000; ++i){
		fread_s(&nLabel, sizeof(unsigned char), sizeof(unsigned char), 1, fp);
		label.mem2d[i][nLabel] = 1.f;

		fread_s(tmp, sizeof(unsigned char) * size, sizeof(unsigned char), size, fp);
		for(int c = 0, cnt = 0; c < 3; ++c){
			for(int y = 0; y < 32; ++y){
				for(int x = 0; x < 32; ++x, ++cnt){
					sample.mem4d[i][c][y][x] = (float)tmp[cnt] / 255.f;
				}
			}
		}
	}
	delete[] tmp;
	fclose(fp);
}