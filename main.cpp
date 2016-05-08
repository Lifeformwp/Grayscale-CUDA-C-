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
#include <vector>
#include "functions.h"

//
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


int main() 
{
	ImgWork *ImageWork = new ImgWork;
	Grayscale *GS = new Grayscale;
	double timeStart_, timeElapsed_;
	int i = 0, k = 0, m = 0;
	int AlgorNum = 0;
	bool repeat = false;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	string str;
	getline(cin, str);
	C_QPCTimer timerq_;
	timerq_.Initialize();
	int ImageAmount = 0;
	vector<Mat> images; 
	vector<Mat> imagesgrey; 
	vector<Mat> imagesgreycpu;
	vector<cudaStream_t> Streams;
	int NumThreadX = deviceProp.maxThreadsPerBlock;

	for (int a=0; a<100;a++) 
	{

		string name = format("C:\img%d.jpg", a+1); 
		Mat img = imread(name); 
		
		if ( img.empty() ) 
			{ 
				cerr << "Successfully loaded " << ImageAmount << " images"<< endl; 
				break; 
			} 
		Mat img3(img.rows,img.cols, CV_8UC1);
		Mat img2(img.rows,img.cols, CV_8UC1);
		
		images.push_back(img); 
		imagesgreycpu.push_back(img2);
		imagesgrey.push_back(img3); 
		
		imshow("Vector of imgs",img);
		waitKey(1000);
		ImageAmount++;
	}
	vector<Mat>::iterator itcpu;
	vector<Mat>::iterator itcpu2;
do{
	cout <<"Input number of algorithm for converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
	cin >>AlgorNum;
	switch(AlgorNum)
	{
	case 1:
		
		for(itcpu=images.begin(), itcpu2=imagesgreycpu.begin(); itcpu!=images.end(),itcpu2!=imagesgreycpu.end(); itcpu++,itcpu2++)
		{

			GS->setData(itcpu, itcpu2, NumThreadX);
			timeStart_ = timerq_.GetTime();
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSLuminosity();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time CPU Luminosity: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
			imshow("CPU Grayscale Luminosity",(*itcpu2));
			waitKey(1000);
			repeat = false;
		}
		break;
		
	case 2:
		for(itcpu=images.begin(), itcpu2=imagesgreycpu.begin(); itcpu!=images.end(),itcpu2!=imagesgreycpu.end(); itcpu++,itcpu2++)
		{
			GS->setData(itcpu, itcpu2, NumThreadX);
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSLightness();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time CPU Lightness: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;			
			imshow("CPU Grayscale Lightness",(*itcpu2));
			waitKey(1000);
			repeat = false;
		}
		break;

	case 3:
		for(itcpu=images.begin(), itcpu2=imagesgreycpu.begin();itcpu!=images.end(),itcpu2!=imagesgreycpu.end();itcpu++,itcpu2++)
		{
			GS->setData(itcpu, itcpu2, NumThreadX);
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSAverage();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time CPU Lightness: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;				
			imshow("CPU Grayscale Average",(*itcpu2));
			waitKey(1000);
			repeat = false;
		}
		break;
	default:
		cout<<"Incorrect input"<<endl;
		repeat = true;
		break;
	}
	}while(repeat == true);
	
	vector<Mat>::iterator it;
	vector<Mat>::iterator it2;
	cudaStream_t* stream = new cudaStream_t[15];
	unsigned char* DevDat = new unsigned char[1024];
	unsigned char* DevOut = new unsigned char[1024];
	for (i; i < ImageAmount;i++) {
                cudaStreamCreate(&stream[i]);
				cudaStreamCreate(&stream[i+1]);
				cudaStreamCreate(&stream[i+2]);
				cudaStreamCreate(&stream[i+3]);
	}


	params par;

	for (k, it=images.begin(), it2=imagesgrey.begin();k<ImageAmount,it!=images.end(),it2!=imagesgrey.end();k++, it++, it2++) {
		ImageWork->setData(it, it2, NumThreadX);
		par.chard = (unsigned char*)stream[k];
		
		
		cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char));
		cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char));

		cudaMemcpyAsync(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice,stream[k+1]);
		cudaStreamSynchronize(stream[k+1]);
		cudaMemcpyAsync(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[k+2]);
		cudaStreamSynchronize(stream[k+2]);
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda1(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
		cudaStreamSynchronize(stream[k]);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		cudaMemcpyAsync(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[k+3]);
		cudaStreamSynchronize(stream[k+3]);
	}

	for (it=images.begin(), it2=imagesgrey.begin();i<ImageAmount,it!=images.end(),it2!=imagesgrey.end();it++, it2++) 
	{
                
			imshow("myWinMemcpy", (*it));
			 waitKey(100);
			  imshow("myWinMemcpy", (*it2));
			 waitKey(100);
	}

	for (m; m < ImageAmount; m++) {
                cudaStreamDestroy(stream[m]);
				cudaStreamDestroy(stream[m+1]);
				cudaStreamDestroy(stream[m+2]);
				cudaStreamDestroy(stream[m+3]);
	}

	//Список картинок
		/*E:\henrik-evensen-castle-valley-v03.jpg
		E:\Assassin.jpg
		C:\Users\Sergey\Documents\Visual Studio 2012\Projects\CudamyExample\3840x2160.bmp*/
		_getch();
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n Grid Size %d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.maxGridSize);

		printf("Input source of image\n Example of right directory file: E:\henrik-evensen-castle-valley-v03.jpg\n Your turn:\n");
		
		cout<< endl << "Your image: " << str << endl;

		//Data for input image
		Mat imgsingle = imread(str, CV_LOAD_IMAGE_COLOR);
		if (imgsingle.empty()) //check whether the image is loaded or not
     {
          cout << "Error : Image cannot be loaded..!!" << endl;
          return -1;
     }
		Mat imgsingle2(imgsingle.rows,imgsingle.cols, CV_8UC1);
		ImageWork->setDataSingle(imgsingle, imgsingle2, NumThreadX);

		printf("Allocating memory on Device\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));

		printf("Copy data on Device\n");
		gpuErrchk(cudaMemcpy(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		
		/*timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
			
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time CPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		*/
		timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
			my_cuda1(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);
		cudaMemcpy(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;

		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);

		namedWindow("Color image", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
		imshow("Color image", imgsingle);

		namedWindow("Gray image", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
		imshow("Gray image", imgsingle2);
		waitKey(0);
		delete ImageWork;
		delete GS;
        return 0;
}
