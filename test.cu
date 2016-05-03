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

extern "C" int my_cuda1(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX)
{
	dim3 blocks(NumBlocksX, 1, 1);
	dim3 threads(NumThreadsX, 1, 1);

	imagegray <<< blocks, threads >>> (devDatIn, devDatOut);

	return 0;
}



