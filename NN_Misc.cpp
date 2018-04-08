#include "NN_Misc.h"



/******************************************************

					Random Genarator

*******************************************************/

float random(float m, float M){
	return (((float)rand() / 32768.0f) * (M - m)) + m;
}


/******************************************************

				   	    Matrix

*******************************************************/

Matrix CreateMat(Dim size, float *data){
	Matrix tmp;

	tmp.nDim = size.dims;
	tmp.d1 = size.d1;
	tmp.d2 = size.d2;
	tmp.d3 = size.d3;
	tmp.d4 = size.d4;

	switch(size.dims){
	case 1:
		tmp.t_elements = (size_t)size.d1;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new float[tmp.t_elements];
			tmp.shared = false;
		}
		break;
	case 2:
		tmp.t_elements = (size_t)size.d1 * size.d2;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new float[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new float*[tmp.d2];
		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2, tmp.d1);
		break;
	case 3:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new float[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new float*[tmp.d2 * tmp.d3];
		tmp.mem3d = new float**[tmp.d3];

		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2 * tmp.d3, tmp.d1);
		JointPtr(tmp.mem3d, tmp.mem2d, tmp.d3, tmp.d2);
		break;
	case 4:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3 * size.d4;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new float[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new float*[tmp.d2 * tmp.d3 * tmp.d4];
		tmp.mem3d = new float**[tmp.d3 * tmp.d4];
		tmp.mem4d = new float***[tmp.d4];

		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2 * tmp.d3 * tmp.d4, tmp.d1);
		JointPtr(tmp.mem3d, tmp.mem2d, tmp.d3 * tmp.d4, tmp.d2);
		JointPtr(tmp.mem4d, tmp.mem3d, tmp.d4, tmp.d3);
		break;
	}

	return tmp;
}

Matrix CloneMat(Matrix &mat){
	Matrix tmp = CreateMat(Dim(mat.d1, mat.d2, mat.d3, mat.d4));

	return tmp;
}

void SetZero(Matrix &mat){
	memset(mat.mem1d, 0, sizeof(float) * mat.t_elements);
}

void ReleaseMat(Matrix &mat){
	if(!mat.shared) delete[] mat.mem1d;
	delete[] mat.mem2d;
	delete[] mat.mem3d;
	delete[] mat.mem4d;
}

void PrintMat(Matrix &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		for(int i = 0; i < mat.d1; ++i) printf("%.1f ", mat.mem1d[i]);
		putchar('\n');
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		for(int i = 0; i < mat.d2; ++i){
			for(int j = 0; j < mat.d1; ++j){
				printf("%.1f ", mat.mem2d[i][j]);
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		for(int i = 0; i < mat.d3; ++i){
			for(int j = 0; j < mat.d2; ++j){
				for(int k = 0; k < mat.d1; ++k){
					printf("%.1f ", mat.mem3d[i][j][k]);
				}
				putchar('\n');
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		for(int i = 0; i < mat.d4; ++i){
			for(int j = 0; j < mat.d3; ++j){
				for(int k = 0; k < mat.d2; ++k){
					for(int m = 0; m < mat.d1; ++m)
						printf("%.1f ", mat.mem4d[i][j][k][m]);
					putchar('\n');
				}
				putchar('\n');
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	}
}

void PrintMatSize(Matrix &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		break;
	}
}


/******************************************************

					     Mark

*******************************************************/

Mark CreateMark(Dim size, int *data){
	Mark tmp;

	tmp.nDim = size.dims;
	tmp.d1 = size.d1;
	tmp.d2 = size.d2;
	tmp.d3 = size.d3;
	tmp.d4 = size.d4;

	switch(size.dims){
	case 1:
		tmp.t_elements = (size_t)size.d1;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new int[tmp.t_elements];
			tmp.shared = false;
		}
		break;
	case 2:		
		tmp.t_elements = (size_t)size.d1 * size.d2;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new int[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new int*[tmp.d2];
		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2, tmp.d1);
		break;
	case 3:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new int[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new int*[tmp.d2 * tmp.d3];
		tmp.mem3d = new int**[tmp.d3];

		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2 * tmp.d3, tmp.d1);
		JointPtr(tmp.mem3d, tmp.mem2d, tmp.d3, tmp.d2);
		break;
	case 4:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3 * size.d4;
		if(data){
			tmp.mem1d = data;
			tmp.shared = true;
		}
		else{
			tmp.mem1d = new int[tmp.t_elements];
			tmp.shared = false;
		}
		tmp.mem2d = new int*[tmp.d2 * tmp.d3 * tmp.d4];
		tmp.mem3d = new int**[tmp.d3 * tmp.d4];
		tmp.mem4d = new int***[tmp.d4];

		JointPtr(tmp.mem2d, tmp.mem1d, tmp.d2 * tmp.d3 * tmp.d4, tmp.d1);
		JointPtr(tmp.mem3d, tmp.mem2d, tmp.d3 * tmp.d4, tmp.d2);
		JointPtr(tmp.mem4d, tmp.mem3d, tmp.d4, tmp.d3);
		break;
	}

	return tmp;
}

void SetZeroMark(Mark &mat){
	memset(mat.mem1d, 0, sizeof(int) * mat.t_elements);
}

void ReleaseMark(Mark &mat){
	if(!mat.shared) delete[] mat.mem1d;
	delete[] mat.mem2d;
	delete[] mat.mem3d;
	delete[] mat.mem4d;
}

void PrintMark(Mark &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		for(int i = 0; i < mat.d1; ++i) printf("%d ", mat.mem1d[i]);
		putchar('\n');
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		for(int i = 0; i < mat.d2; ++i){
			for(int j = 0; j < mat.d1; ++j){
				printf("%d ", mat.mem2d[i][j]);
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		for(int i = 0; i < mat.d3; ++i){
			for(int j = 0; j < mat.d2; ++j){
				for(int k = 0; k < mat.d1; ++k){
					printf("%d ", mat.mem3d[i][j][k]);
				}
				putchar('\n');
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		for(int i = 0; i < mat.d4; ++i){
			for(int j = 0; j < mat.d3; ++j){
				for(int k = 0; k < mat.d2; ++k){
					for(int m = 0; m < mat.d1; ++m)
						printf("%d ", mat.mem4d[i][j][k][m]);
					putchar('\n');
				}
				putchar('\n');
			}
			putchar('\n');
		}
		putchar('\n');
		break;
	}
}

void PrintMarkSize(Mark &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		break;
	}
}


/******************************************************

					    GMatrix

*******************************************************/

GMatrix CreateGMat(Dim size, float *data){
	GMatrix tmp;

	tmp.d1 = size.d1;
	tmp.d2 = size.d2;
	tmp.d3 = size.d3;
	tmp.d4 = size.d4;
	tmp.nDim = size.dims;

	switch(size.dims){
	case 1:
		tmp.t_elements = (size_t)size.d1;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(float) * tmp.t_elements));
			tmp.shared = false;
		}
		break;
	case 2:
		tmp.t_elements = (size_t)size.d1 * size.d2;
		
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(float) * tmp.t_elements));
			tmp.shared = false;
		}
		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(float*) * tmp.d2));
		ErrorChk(SetGMat2D(tmp.gmem2d, tmp.gmem1d, tmp.d2, tmp.d1));

		break;
	case 3:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(float) * tmp.t_elements));
			tmp.shared = false;
		}
		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(float*) * tmp.d2 * tmp.d3));
		ErrorChk(cudaMalloc(&tmp.gmem3d, sizeof(float**) * tmp.d3));
		ErrorChk(SetGMat2D(tmp.gmem2d, tmp.gmem1d, tmp.d2 * tmp.d3, tmp.d1));
		ErrorChk(SetGMat3D(tmp.gmem3d, tmp.gmem2d, tmp.d3, tmp.d2));

		break;
	case 4:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3 * size.d4;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(float) * tmp.t_elements));
			tmp.shared = false;
		}
		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(float*) * tmp.d2 * tmp.d3 * tmp.d4));
		ErrorChk(cudaMalloc(&tmp.gmem3d, sizeof(float**) * tmp.d3 * tmp.d4));
		ErrorChk(cudaMalloc(&tmp.gmem4d, sizeof(float***) * tmp.d4));
		ErrorChk(SetGMat2D(tmp.gmem2d, tmp.gmem1d, tmp.d2 * tmp.d3 * tmp.d4, tmp.d1));
		ErrorChk(SetGMat3D(tmp.gmem3d, tmp.gmem2d, tmp.d3 * tmp.d4, tmp.d2));
		ErrorChk(SetGMat4D(tmp.gmem4d, tmp.gmem3d, tmp.d4, tmp.d3));
	}

	return tmp;
}

GMatrix CloneGMat(GMatrix &mat){
	GMatrix tmp = CreateGMat(Dim(mat.d1, mat.d2, mat.d3, mat.d4));

	return tmp;
}

void ZeroGMat(GMatrix &mat){
	ErrorChk(cudaMemset(mat.gmem1d, 0, mat.t_elements * sizeof(float)));
}

void ReleaseGMat(GMatrix &mat){
	if(!mat.shared) cudaFree(mat.gmem1d);
	cudaFree(mat.gmem2d);
	cudaFree(mat.gmem3d);
	cudaFree(mat.gmem4d);
}


void PrintGMatSize(GMatrix &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		break;
	}
}


/******************************************************

					     GMark

*******************************************************/

GMark CreateGMark(Dim size, int *data){
	GMark tmp;

	tmp.nDim = size.dims;
	tmp.d1 = size.d1;
	tmp.d2 = size.d2;
	tmp.d3 = size.d3;
	tmp.d4 = size.d4;

	switch(size.dims){
	case 1:
		tmp.t_elements = (size_t)size.d1;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(int) * tmp.t_elements));
			tmp.shared = false;
		}
		break;
	case 2:
		tmp.t_elements = (size_t)size.d1 * size.d2;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(int) * tmp.t_elements));
			tmp.shared = false;
		}

		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(int*) * tmp.d2));
		ErrorChk(SetGMark2D(tmp.gmem2d, tmp.gmem1d, tmp.d2, tmp.d1));
		break;
	case 3:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(int) * tmp.t_elements));
			tmp.shared = false;
		}

		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(int*) * tmp.d2 * tmp.d3));
		ErrorChk(cudaMalloc(&tmp.gmem3d, sizeof(int**) * tmp.d3));
		ErrorChk(SetGMark2D(tmp.gmem2d, tmp.gmem1d, tmp.d2 * tmp.d3, tmp.d1));
		ErrorChk(SetGMark3D(tmp.gmem3d, tmp.gmem2d, tmp.d3, tmp.d2));
		break;
	case 4:
		tmp.t_elements = (size_t)size.d1 * size.d2 * size.d3 * size.d4;
		if(data){
			tmp.gmem1d = data;
			tmp.shared = true;
		}
		else{
			ErrorChk(cudaMalloc(&tmp.gmem1d, sizeof(int) * tmp.t_elements));
			tmp.shared = false;
		}
			
		ErrorChk(cudaMalloc(&tmp.gmem2d, sizeof(int*) * tmp.d2 * tmp.d3 * tmp.d4));
		ErrorChk(cudaMalloc(&tmp.gmem3d, sizeof(int**) * tmp.d3 * tmp.d4));
		ErrorChk(cudaMalloc(&tmp.gmem4d, sizeof(int***) * tmp.d4));
		ErrorChk(SetGMark2D(tmp.gmem2d, tmp.gmem1d, tmp.d2 * tmp.d3 * tmp.d4, tmp.d1));
		ErrorChk(SetGMark3D(tmp.gmem3d, tmp.gmem2d, tmp.d3 * tmp.d4, tmp.d2));
		ErrorChk(SetGMark4D(tmp.gmem4d, tmp.gmem3d, tmp.d4, tmp.d3));
		break;
	}

	return tmp;
}

void ZeroGMark(GMark &mat){
	ErrorChk(cudaMemset(mat.gmem1d, 0, sizeof(int) * mat.t_elements));
}

void ReleaseGMark(GMark &mat){
	if(!mat.shared) cudaFree(mat.gmem1d);
	cudaFree(mat.gmem2d);
	cudaFree(mat.gmem3d);
	cudaFree(mat.gmem4d);
}

void PrintGMarkSize(GMark &mat){
	switch(mat.nDim){
	case 1:
		printf("[%d]\n", mat.d1);
		break;
	case 2:
		printf("[%d x %d]\n", mat.d1, mat.d2);
		break;
	case 3:
		printf("[%d x %d x %d]\n", mat.d1, mat.d2, mat.d3);
		break;
	case 4:
		printf("[%d x %d x %d x %d]\n", mat.d1, mat.d2, mat.d3, mat.d4);
		break;
	}
}



/******************************************************

					 Convert Matrix

*******************************************************/

void MemCpyH2D(Matrix &src, GMatrix &dst){
	if(src.t_elements != dst.t_elements){
		cout << "행렬 원소 개수가 틀립니다.\n";
		return;
	}
	ErrorChk(cudaMemcpy(dst.gmem1d, src.mem1d, sizeof(float) * dst.t_elements, cudaMemcpyHostToDevice));
}

void MemCpyD2H(GMatrix &src, Matrix &dst){
	if(src.t_elements != dst.t_elements){
		cout << "행렬 원소 개수가 틀립니다.\n";
		return;
	}
	ErrorChk(cudaMemcpy(dst.mem1d, src.gmem1d, sizeof(float) * dst.t_elements, cudaMemcpyDeviceToHost));
}