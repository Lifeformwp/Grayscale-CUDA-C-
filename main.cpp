#include <iostream>
#include <opencv2/opencv.hpp>
#include <conio.h>
#include <string.h>
#include "QPCTimer.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "test.h"


using namespace std;
using namespace cv;
#define CYCLES 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{

   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void faktorial(int InSize, unsigned char *DataIn, unsigned char *DataOut)// заголовок функции
{
	for(int i = 0,  j = 0; i < InSize; i += 3, j++)
	{
		DataOut[j] = (DataIn[i] + DataIn[i + 1] + DataIn[i + 2]) / 3;
	}

}


int main() 
{
		
		
		C_QPCTimer timerq_;
		timerq_.Initialize();

		double timeStart_, timeElapsed_;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		cudaStream_t streamA, streamB;
		cudaEvent_t eventA, eventB;
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n Grid Size %d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.maxGridSize);
		char* c = "";
		printf("Input source of image\n Example of right directory file: E:\henrik-evensen-castle-valley-v03.jpg\n Your turn:\n");
		string str;
		getline(cin, str);
		cout<< endl << "Your image: " << str << endl;

		//Data for input image
		Mat img = imread(str, CV_LOAD_IMAGE_COLOR);
		if (img.empty()) //check whether the image is loaded or not
     {
          cout << "Error : Image cannot be loaded..!!" << endl;
          //system("pause"); //wait for a key press
          return -1;
     }
		
		unsigned char* DataImg = img.data;
		Mat img3(img.rows,img.cols, CV_8UC1);
		int NumThreadsX = deviceProp.maxThreadsPerBlock;
		int NumBlocksX = (img.cols * img.rows)/NumThreadsX;
		unsigned char* DataImg2 = (unsigned char*)img3.data;
		int step1 = img.step;
		int step2 = img3.step;
		params par;
		par.DatIn = DataImg;
		par.DatOut = DataImg2;
		par.NumBlocksX = NumBlocksX;
		par.NumThreadsX = NumThreadsX;
		int SizeIn = (step1*img.rows);
		int SizeOut = (step2*img3.rows);
		par.SizeIn = SizeIn;
		par.SizeOut = SizeOut;

		printf("Allocating memory on Device\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char)));

		printf("Copy data on Device\n");
		gpuErrchk(cudaMemcpy(par.devDatIn, par.DatIn, par.SizeIn * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, par.DatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice));

		timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
			faktorial(SizeIn, DataImg, DataImg2);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;

		cout << "Average time CPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		
		timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
		my_cuda1(par.devDatIn, par.devDatOut, par.NumBlocksX, par.NumThreadsX);
		gpuErrchk(cudaMemcpy(par.DatOut, par.devDatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;

		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);

		namedWindow("Color image", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
		imshow("Color image", img);

		namedWindow("Gray image", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
		imshow("Gray image", img3);
		waitKey(0);
    return 0;
}
