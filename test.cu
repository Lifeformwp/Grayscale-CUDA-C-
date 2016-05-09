#include <stdio.h>
#include "test.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "conio.h"
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void imagegray(unsigned char *DataIn, unsigned char *DataOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    DataOut[idx] = (unsigned int)((DataIn[3*idx] + DataIn[3*idx+1] + DataIn[3*idx+2])/3);
}

__global__ void imagegraylumin(unsigned char *DataIn, unsigned char *DataOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    DataOut[idx] = (unsigned int)((0.21 * DataIn[3*idx]) + (0.72 * DataIn[3*idx+1]) + (0.07 * DataIn[3*idx+2]));
}

__global__ void imagegraylightn(unsigned char *DataIn, unsigned char *DataOut)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int min= DataIn[3*idx]<DataIn[3*idx + 1]?DataIn[3*idx]:DataIn[3*idx + 1];
	min= DataIn[3*idx + 2]<min?DataIn[3*idx + 2]:min;
	int max = DataIn[3*idx]<DataIn[3*idx + 1]?DataIn[3*idx]:DataIn[3*idx + 1];
	max=DataIn[3*idx + 2]<max?DataIn[3*idx + 2]:max;
	DataOut[idx] = (max + min)/2;
    
}


extern "C" int my_cuda1(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX, unsigned char *chard)
{
	dim3 blocks(NumBlocksX, 1, 1);
	dim3 threads(NumThreadsX, 1, 1);
	cudaStream_t stream = (cudaStream_t)chard;
	imagegraylumin <<< blocks, threads, 0, stream >>> (devDatIn, devDatOut);

	return 0;
}

extern "C" int my_cuda2(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX, unsigned char *chard)
{
	dim3 blocks(NumBlocksX, 1, 1);
	dim3 threads(NumThreadsX, 1, 1);
	cudaStream_t stream = (cudaStream_t)chard;
	imagegraylightn <<< blocks, threads, 0, stream >>> (devDatIn, devDatOut);

	return 0;
}

extern "C" int my_cuda3(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX, unsigned char *chard)
{
	dim3 blocks(NumBlocksX, 1, 1);
	dim3 threads(NumThreadsX, 1, 1);
	cudaStream_t stream = (cudaStream_t)chard;
	imagegray <<< blocks, threads, 0, stream >>> (devDatIn, devDatOut);

	return 0;
}
