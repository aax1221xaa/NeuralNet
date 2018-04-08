#ifndef CIFAR_10_H
#define CIFAR_10_H

#include "NN_Misc.h"

class CIFAR_10{
public:
	Matrix sample;
	Matrix label;
	char name[10][20];

	~CIFAR_10();
	CIFAR_10(const char dir_path[], int nFile);
	CIFAR_10(const char dir_path[]);

	void LoadMataFile(const char dir_path[]);
	void LoadSampleFile(const char dir_path[]);
};

#endif