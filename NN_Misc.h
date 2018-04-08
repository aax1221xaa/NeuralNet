#ifndef _NN_MISC_H
#define _NN_MISC_H

#include <iostream>
#include "_NN_Misc.cuh"

using namespace std;

/******************************************************

					Random Genarator

*******************************************************/

float random(float m, float M);



/******************************************************

					   ErrorChk

*******************************************************/


#define ErrorChk(ans) { _ErrorCheck((ans), __FILE__, __LINE__); }


inline void _ErrorCheck(cudaError_t code, char *file, int line, bool abort = true){
	if(cudaSuccess != code){
		cout << "GPU Error : " << cudaGetErrorString(code) << ' ' << file << ' ' << line << endl; 
		if(abort) exit(code);
	}
}


/******************************************************

					   DImension

*******************************************************/

struct Dim{
	int d1, d2, d3, d4;
	int dims;

	Dim(){
		Clear();
	}

	Dim(int _d1, int _d2 = 0, int _d3 = 0, int _d4 = 0){
		int dim[4] = {_d1, _d2, _d3, _d4};
		bool _error = false;
		int i = 0;

		for(i = 0; i < 4; ++i){
			if(dim[i] < 0){
				cout << "디멘젼 크기를 음수로 만들 수 없습니다." << endl;
				_error = true;
				Clear();
				break;
			}
		}
		for(i = 0; i < 3 && !_error; ++i){
			if(dim[i] == 0 && dim[i + 1]){
				cout << "디멘젼 크기가 잘 못 되었습니다." << endl;
				Clear();
				_error = true;
				break;
			}
		}
		if(!_error){
			d1 = _d1;
			d2 = _d2;
			d3 = _d3;
			d4 = _d4;
			for(i = 0; i < 4; ++i){
				if(dim[i] == 0) break;
			}
			if(i == 4) dims = 4;
			else dims = i;
		}
	}

	void Clear(){
		d1 = d2 = d3 = d4 = 0;
		dims = 0;
	}
};


/******************************************************

					     Matrix

*******************************************************/

struct Matrix{
	float *mem1d;
	float **mem2d;
	float ***mem3d;
	float ****mem4d;

	int d1;
	int d2;
	int d3;
	int d4;
	int nDim;
	size_t t_elements;

	bool shared;

	Matrix(){
		mem1d = NULL;
		mem2d = NULL;
		mem3d = NULL;
		mem4d = NULL;

		d1 = d2 = d3 = d4 = nDim = 0;
		t_elements = 0;

		shared = false;
	}
};

template <typename _T>
void JointPtr(_T *dst, _T src, int iter, int step){
	for(int i = 0; i < iter; ++i) dst[i] = src + i * step;
}

Matrix CreateMat(Dim size, float *data = NULL);
Matrix CloneMat(Matrix &mat);
void SetZero(Matrix &mat);
void ReleaseMat(Matrix &mat);
void PrintMat(Matrix &mat);
void PrintMatSize(Matrix &mat);


/******************************************************

					     Mark

*******************************************************/

struct Mark{
	int *mem1d;
	int **mem2d;
	int ***mem3d;
	int ****mem4d;

	int d1;
	int d2;
	int d3;
	int d4;
	int nDim;
	size_t t_elements;

	bool shared;

	Mark(){
		mem1d = NULL;
		mem2d = NULL;
		mem3d = NULL;
		mem4d = NULL;

		d1 = d2 = d3 = d4 = nDim = 0;
		t_elements = 0;

		shared = false;
	}
};

Mark CreateMark(Dim size, int *data = NULL);
void SetZeroMark(Mark &mat);
void ReleaseMark(Mark &mat);
void PrintMark(Mark &mat);
void PrintMarkSize(Mark &mat);


/******************************************************

					    GMatrix

*******************************************************/

struct GMatrix{
	float *gmem1d;
	float **gmem2d;
	float ***gmem3d;
	float ****gmem4d;

	int d1;
	int d2;
	int d3;
	int d4;
	int nDim;
	size_t t_elements;

	bool shared;

	GMatrix(){
		gmem1d = NULL;
		gmem2d = NULL;
		gmem3d = NULL;
		gmem4d = NULL;

		d1 = d2 = d3 = d4 = nDim = 0;
		t_elements = 0;

		shared = false;
	}
};

GMatrix CreateGMat(Dim size, float *data = NULL);
GMatrix CloneGMat(GMatrix &mat);
void ZeroGMat(GMatrix &mat);
void ReleaseGMat(GMatrix &mat);
void PrintGMatSize(GMatrix &mat);


/******************************************************

					     GMark

*******************************************************/

struct GMark{
	int *gmem1d;
	int **gmem2d;
	int ***gmem3d;
	int ****gmem4d;

	int d1;
	int d2;
	int d3;
	int d4;
	int nDim;
	size_t t_elements;

	bool shared;

	GMark(){
		gmem1d = NULL;
		gmem2d = NULL;
		gmem3d = NULL;
		gmem4d = NULL;

		d1 = d2 = d3 = d4 = nDim = 0;
		t_elements = 0;

		shared = false;
	}
};

GMark CreateGMark(Dim size, int *data = NULL);
void ZeroGMark(GMark &mat);
void ReleaseGMark(GMark &mat);
void PrintGMarkSize(GMark &mat);


/******************************************************

					 Convert Matrix

*******************************************************/

void MemCpyH2D(Matrix &src, GMatrix &dst);
void MemCpyD2H(GMatrix &src, Matrix &dst);



/******************************************************

						GMatxD

*******************************************************/

struct GMat1D{
	float *data;
	int d1;
};

struct GMat2D{
	float **data;
	int d1;
	int d2;
};

struct GMat3D{
	float ***data;
	int d1;
	int d2;
	int d3;
};

struct GMat4D{
	float ****data;
	int d1;
	int d2;
	int d3;
	int d4;
};


#endif