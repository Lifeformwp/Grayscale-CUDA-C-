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
#define IMAGE_NUM 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{

   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//void faktorial(int InSize, unsigned char *DataIn, unsigned char *DataOut)// заголовок функции
//{
//	for(int i = 0,  j = 0; i < InSize; i += 3, j++)
//	{
//		DataOut[j] = (DataIn[i] + DataIn[i + 1] + DataIn[i + 2]) / 3;
//	}
//
//}


int main() 
{

	BlackWhiteImg	*ClImage  = new BlackWhiteImg;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	/*cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);*/
	vector<Mat> images; 
	vector<Mat> imagesgrey; 
	int Count=4; 
	for (int a=0; a<Count;a++) 
	{

		string name = format("C:\img%d.jpg", a+1); 
		Mat img = imread(name); 
		
		if ( img.empty() ) 
			{ 
				cerr << "\nERROR: Can't be loaded image" << name << endl; 
				continue; 
			} 
		Mat img3(img.rows,img.cols, CV_8UC1);
		int SizeInImg = img.step * img.rows;
		unsigned char* DataImg = img.data;
		unsigned char* DataImg2 = img3.data;
		/*int NumThreadsX = deviceProp.maxThreadsPerBlock;
		int NumBlocksX = (img.cols * img.rows)/NumThreadsX*/;
		ClImage->setData(SizeInImg, DataImg2, DataImg);
		ClImage->Grayscale();
		ClImage->getGrayscaleData();
		//faktorial(SizeInImg, DataImg, DataImg2);
		images.push_back(img); 
		imshow("Vector of imgs",img);
		cvWaitKey(1000);
		imagesgrey.push_back(img3); 
		/*params par;
		par.DatIn = DataImg;
		par.DatOut = DataImg2;
		par.NumBlocksX = NumBlocksX;
		par.NumThreadsX = NumThreadsX;
		int SizeIn = (img.step*img.rows);
		int SizeOut = (img3.step*img3.rows);
		par.SizeIn = SizeIn;
		par.SizeOut = SizeOut;

		printf("Allocating memory on Device\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char)));

		printf("Copy data on Device\n");
		gpuErrchk(cudaMemcpy(par.devDatIn, par.DatIn, par.SizeIn * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, par.DatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice));*/
		imshow("Vector of imgs",img3);
		cvWaitKey(1000); 
	}

	
	 vector<Mat>::iterator it;
	 vector<Mat>::iterator it2;
	 vector<Mat>::iterator it3;
	 vector<Mat>::iterator it4;
	 
	 for (it = images.begin(); it != images.end(); ++it)
	 {

		 imshow("MYSHOW", (*it));
		 waitKey(100);
	 }
	 it = images.begin();
	 it2 = images.begin()+1;
	 it3 = images.begin()+2;
	 it4 = images.begin()+3;

	 unsigned char  *devDatInFirst;
	unsigned char *devDatOutFirst;
	Mat img4((*it).rows,(*it).cols, CV_8UC1);
	Mat img5((*it2).rows,(*it2).cols, CV_8UC1);
	Mat img6((*it3).rows,(*it3).cols, CV_8UC1);
	Mat img7((*it4).rows,(*it4).cols, CV_8UC1);
	//int SizeInFirst = (*it).step * (*it).rows;
	//int SizeOutFirst = img4.step * img4.rows;
	//int SizeInFirst2 = (*it2).step * (*it2).rows;
	//int SizeOutFirst2 = img5.step * img5.rows;
	//int SizeInFirst3 = (*it3).step * (*it3).rows;
	//int SizeOutFirst3 = img6.step * img6.rows;
	//int SizeInFirst4 = (*it4).step * (*it4).rows;
	//int SizeOutFirst4 = img7.step * img7.rows;
	unsigned char *DatInFirst = (*it).data;
	unsigned char* DatOutFirst = img4.data;
	unsigned char *DatInFirst2 = (*it2).data;
	unsigned char* DatOutFirst2 = img5.data;
	unsigned char *DatInFirst3 = (*it3).data;
	unsigned char* DatOutFirst3 = img6.data;
	unsigned char *DatInFirst4 = (*it4).data;
	unsigned char* DatOutFirst4 = img7.data;
	int NumThreadsX = deviceProp.maxThreadsPerBlock;
	int NumBlocksX = ((*it).cols * (*it).rows)/NumThreadsX;
	int NumThreadsX2 = deviceProp.maxThreadsPerBlock;
	int NumBlocksX2 = ((*it2).cols * (*it2).rows)/NumThreadsX;
	int NumThreadsX3 = deviceProp.maxThreadsPerBlock;
	int NumBlocksX3 = ((*it3).cols * (*it3).rows)/NumThreadsX;
	int NumThreadsX4 = deviceProp.maxThreadsPerBlock;
	int NumBlocksX4 = ((*it4).cols * (*it4).rows)/NumThreadsX;
		params par;
		par.DatIn = DatInFirst;
		par.DatOut = DatOutFirst;
		par.DatIn2 = DatInFirst2;
		par.DatOut2 = DatOutFirst2;
		par.DatIn3 = DatInFirst3;
		par.DatOut3 = DatOutFirst3;
		par.DatIn4 = DatInFirst4;
		par.DatOut4 = DatOutFirst4;
		par.NumBlocksX = NumBlocksX;
		par.NumThreadsX = NumThreadsX;
		par.NumBlocksX2 = NumBlocksX2;
		par.NumThreadsX2 = NumThreadsX2;
		par.NumBlocksX3 = NumBlocksX3;
		par.NumThreadsX3 = NumThreadsX3;
		par.NumBlocksX4 = NumBlocksX4;
		par.NumThreadsX4 = NumThreadsX4;
		int SizeIn = ((*it).step*(*it).rows);
		int SizeOut = (img4.step*img4.rows);
		int SizeIn2 = ((*it2).step*(*it2).rows);
		int SizeOut2 = (img5.step*img5.rows);
		int SizeIn3 = ((*it3).step*(*it3).rows);
		int SizeOut3 = (img6.step*img6.rows);
		int SizeIn4 = ((*it4).step*(*it4).rows);
		int SizeOut4 = (img7.step*img7.rows);
		par.SizeIn = SizeIn;
		par.SizeOut = SizeOut;
		par.SizeIn2 = SizeIn2;
		par.SizeOut2 = SizeOut2;
		par.SizeIn3 = SizeIn3;
		par.SizeOut3 = SizeOut3;
		par.SizeIn4 = SizeIn4;
		par.SizeOut4 = SizeOut4;
		unsigned char* chard = (unsigned char*)stream1;
		unsigned char* chard1 = (unsigned char*)stream2;
		unsigned char* chard2 = (unsigned char*)stream3;
		unsigned char* chard3 = (unsigned char*)stream4;
		par.chard = chard;
		par.chard1 = chard1;
		par.chard2 = chard2;
		par.chard3 = chard3;
		//First
		printf("Allocating memory on Device image 1\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char)));

		printf("Copy data on Device image 1\n");
		gpuErrchk(cudaMemcpyAsync(par.devDatIn, par.DatIn, par.SizeIn * sizeof(unsigned char), cudaMemcpyHostToDevice,stream1));
		gpuErrchk(cudaMemcpyAsync(par.devDatOut, par.DatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice,stream1));

		//Second
		printf("Allocating memory on Device image 2\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn2, par.SizeIn2 * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut2, par.SizeOut2 * sizeof(unsigned char)));

		printf("Copy data on Device image 2\n");
		gpuErrchk(cudaMemcpyAsync(par.devDatIn2, par.DatIn2, par.SizeIn2 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream2));
		gpuErrchk(cudaMemcpyAsync(par.devDatOut2, par.DatOut2, par.SizeOut2 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream2));

		//Third
		printf("Allocating memory on Device image 3\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn3, par.SizeIn3 * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut3, par.SizeOut3 * sizeof(unsigned char)));

		printf("Copy data on Device image 3\n");
		gpuErrchk(cudaMemcpyAsync(par.devDatIn3, par.DatIn3, par.SizeIn3 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream3));
		gpuErrchk(cudaMemcpyAsync(par.devDatOut3, par.DatOut3, par.SizeOut3 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream3));

		//Fourth
		printf("Allocating memory on Device image 4\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn4, par.SizeIn4 * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut4, par.SizeOut4 * sizeof(unsigned char)));

		printf("Copy data on Device image 4\n");
		gpuErrchk(cudaMemcpyAsync(par.devDatIn4, par.DatIn4, par.SizeIn4 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream4));
		gpuErrchk(cudaMemcpyAsync(par.devDatOut4, par.DatOut4, par.SizeOut4 * sizeof(unsigned char), cudaMemcpyHostToDevice,stream4));

	 //		printf("Allocating memory on Device\n");
		//cudaMalloc((void**)&devDatInFirst, SizeInFirst * sizeof(unsigned char));
		//cudaMalloc((void**)&devDatOutFirst, SizeOutFirst * sizeof(unsigned char));

		//printf("Copy data on Device\n");
		//cudaMemcpy(devDatInFirst, DatInFirst, SizeInFirst * sizeof(unsigned char), cudaMemcpyHostToDevice);
		//cudaMemcpy(devDatOutFirst, DatOutFirst, SizeOutFirst * sizeof(unsigned char), cudaMemcpyHostToDevice);

		//printf("Writing an output image.\n");
		//cudaMemcpyAsync(DatOutFirst, devDatOutFirst, SizeOutFirst * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		my_cuda1(par.devDatIn, par.devDatOut, par.NumBlocksX, par.NumThreadsX, par.chard);
		my_cuda1(par.devDatIn2, par.devDatOut2, par.NumBlocksX2, par.NumThreadsX2, par.chard1);
		my_cuda1(par.devDatIn3, par.devDatOut3, par.NumBlocksX3, par.NumThreadsX3, par.chard2);
		my_cuda1(par.devDatIn4, par.devDatOut4, par.NumBlocksX4, par.NumThreadsX4, par.chard3);

		gpuErrchk(cudaMemcpyAsync(par.DatOut, par.devDatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream1));
		gpuErrchk(cudaMemcpyAsync(par.DatOut2, par.devDatOut2, par.SizeOut2 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream2));
		gpuErrchk(cudaMemcpyAsync(par.DatOut3, par.devDatOut3, par.SizeOut3 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream3));
		gpuErrchk(cudaMemcpyAsync(par.DatOut4, par.devDatOut4, par.SizeOut4 * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream4));
		printf("Copying Done\n");

		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);
		cudaFree(par.devDatOut2);
		cudaFree(par.devDatIn2);
		cudaFree(par.devDatOut3);
		cudaFree(par.devDatIn3);
		cudaFree(par.devDatOut4);
		cudaFree(par.devDatIn4);
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
		cudaStreamDestroy(stream3);
		cudaStreamDestroy(stream4);

	imshow("myWin2", (*it));
	 waitKey(0);
	 imshow("myWin2", img4);
	 waitKey(0);
	 imshow("myWin2", (*it2));
	 waitKey(0);
	  imshow("myWin2", img5);
	 waitKey(0);
	 imshow("myWin2", (*it3));
	 waitKey(0);
	  imshow("myWin2", img6);
	 waitKey(0);
	 imshow("myWin2", (*it4));
	 waitKey(0);
	  imshow("myWin2", img7);
	 waitKey(0);

    for (it = imagesgrey.begin(); it != imagesgrey.end() ; it++) {
        imshow("myWin", (*it));
        waitKey(100);
    }
	cout << images.size()<<endl;
		
		
		C_QPCTimer timerq_;
		timerq_.Initialize();

		double timeStart_, timeElapsed_;
		/*cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);*/
		cudaStream_t streamA, streamB;
		cudaEvent_t eventA, eventB;
		//string name = "E:\Assassin.jpg";
//	/*int i=0;*/
//	Mat gray;
//	
//	/*while(b){  */
//
//	for (int i=0; i<4; i++)
//	{
//
//    //sprintf(name, "img%d.jpg",i);
//    Mat src= imread(name,1);
//	namedWindow("src", CV_WINDOW_AUTOSIZE);
//	imshow("src",src);
//    // if(!src.data ) break;
//
//     //cvtColor(src,gray,CV_BGR2GRAY);
//     //sprintf(name, "gray%d.jpg\n\n",i);
//     imwrite(name, gray);
//
//    imshow("src",src);
//    imshow("result",gray);
//
//    /*i++;*/
//    waitKey();
//}
//}

																	//Список картинок
		/*E:\henrik-evensen-castle-valley-v03.jpg
		E:\Assassin.jpg
		C:\Users\Sergey\Documents\Visual Studio 2012\Projects\CudamyExample\3840x2160.bmp*/
		printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n Grid Size %d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.maxGridSize);
		char* c = "";
		printf("Input source of image\n Example of right directory file: E:\henrik-evensen-castle-valley-v03.jpg\n Your turn:\n");
		/*char *tbLEN;
		tbLEN = new char [1024];*/
		string str;
		/*cin.getline(tbLEN,1024);
*/
	
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
	/*	int NumThreadsX = deviceProp.maxThreadsPerBlock;
		int NumBlocksX = (img.cols * img.rows)/NumThreadsX;*/
		unsigned char* DataImg2 = (unsigned char*)img3.data;
		int step1 = img.step;
		int step2 = img3.step;
		//params par;
		//par.DatIn = DataImg;
		//par.DatOut = DataImg2;
		//par.NumBlocksX = NumBlocksX;
		//par.NumThreadsX = NumThreadsX;
		//int SizeIn = (step1*img.rows);
		//int SizeOut = (step2*img3.rows);
		//par.SizeIn = SizeIn;
		//par.SizeOut = SizeOut;

		printf("Allocating memory on Device\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char)));

		printf("Copy data on Device\n");
		gpuErrchk(cudaMemcpy(par.devDatIn, par.DatIn, par.SizeIn * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, par.DatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice));

	/*	ClImage->setData(SizeIn, DataImg, DataImg2);
		
		timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
			ClImage->Grayscale();
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		ClImage->getGrayscaleData();
		cout << "Average time CPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;*/
		
		//faktorial(SizeIn, DatIn, DatOut);
		timeStart_ = timerq_.GetTime();
		for (int j = 0; j < CYCLES; j++)
		{
//		my_cuda1(par.devDatIn, par.devDatOut, par.NumBlocksX, par.NumThreadsX);
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

		
		//IplImage* image;
		//image = cvLoadImage(tbLEN, 1);
		//int height = image->height;
		//int width = image->width;
		//int step = image->widthStep;
	 //   int SizeIn = (step*height);
		//
		//printf("\nProcessing image\n");
		//int Width = img.cols;
		//int Height = img.rows;
		////int Step = img.step;
		//int SizeInImg = step * Height;
//		Mat img2 = cvCreateImage(cvSize(Width, Height), IPL_DEPTH_8U, 1);
		
		
		//faktorial(SizeInImg, DataImg, DataImg2);
		

		//namedWindow("MyWindow", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
		//imshow("MyWindow", img3);

		////Data for output image
		//IplImage *image2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		//int step2 = image2->widthStep;
		//int SizeOut = step2 * height;

		////GPU
		//unsigned char* DatIn = (unsigned char*)image->imageData;
		//unsigned char* DatOut = (unsigned char*)image2->imageData;

		//int NumThreadsX = deviceProp.maxThreadsPerBlock;
		//int NumBlocksX = (width * height)/NumThreadsX;
		//params par;
		//par.DatIn = DatIn;
		//par.DatOut = DatOut;
		//par.SizeIn = SizeIn;
		//par.SizeOut = SizeOut;
		//par.NumBlocksX = NumBlocksX;
		//par.NumThreadsX = NumThreadsX;

		//par.SizeOut = SizeOut;

		//printf("Allocating memory on Device\n");
		///* Allocate memory on Device */
		//gpuErrchk(cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(unsigned char)));
		//gpuErrchk(cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char)));

		//printf("Copy data on Device\n");
	 //   /* Copy data on Device */
		//gpuErrchk(cudaMemcpy(par.devDatIn, par.DatIn, par.SizeIn * sizeof(unsigned char), cudaMemcpyHostToDevice));
		//gpuErrchk(cudaMemcpy(par.devDatOut, par.DatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice));

		//timeStart_ = timerq_.GetTime();
		//for (int j = 0; j < CYCLES; j++)
		//{
		//	faktorial(SizeIn, DatIn, DatOut);
		//}
		//timeElapsed_ = timerq_.GetTime() - timeStart_;

		//cout << "Average time CPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		//
		////faktorial(SizeIn, DatIn, DatOut);
		//timeStart_ = timerq_.GetTime();
		//for (int j = 0; j < CYCLES; j++)
		//{
		//	my_cuda1(par.devDatIn, par.devDatOut, par.NumBlocksX, par.NumThreadsX);
		//	gpuErrchk(cudaMemcpy(par.DatOut, par.devDatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		//}
		//
		//timeElapsed_ = timerq_.GetTime() - timeStart_;


		//cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;

		//

		//cudaFree(par.devDatOut);
		//cudaFree(par.devDatIn);


		//cvNamedWindow("Imagecolor");
		//cvShowImage("Imagecolor", image);

		//cvNamedWindow("gray");
		//cvShowImage("gray", image2);

		//const char* filename1 = "CcPwSwMW4AELPUc.jpg";
		//printf("Saving an output image\n");
		//cvSaveImage( filename1, image2 );

        /*cvWaitKey(0);*/
		delete ClImage;
        return 0;
}







/*Предыдущий Separate

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "conio.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include "VPImgFiles.h"
//#include "test.h"
//#include "QPCTimer.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//
//using namespace cinegy::vp::image::img_files;
//
//using namespace std;
//
//#define CYCLES 100
//
//void my_cuda_func3(params *pParams)
//{
//	C_QPCTimer timerq_;
//	timerq_.Initialize();	
//	double timeStart_, timeElapsed_;
//	printf("Processing writing the picture data in file for %d iterations\n", CYCLES);
//	timeStart_ = timerq_.GetTime();
//	for (int j = 0; j < CYCLES; j++)
//	{
//		WriteBmpFile(L"3840x2160_gray.bmp", pParams->iWidth, pParams->iHeight, 8, pParams->SizeOut * sizeof(unsigned char), pParams->POutData, false);
//	}
//	timeElapsed_ = timerq_.GetTime() - timeStart_;
//
//	cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl<<"Done.";
//	_getch();
//}
//
//
//void faktorial(int InSize, unsigned char *DataIn, unsigned char *DataOut, int WidthI, int HeightI)// заголовок функции
//{
//	for(int i = 0,  j = 0; i < InSize; i += 4, j++)
//	{
//		DataOut[j] = (DataIn[i] + DataIn[i + 1] + DataIn[i + 2]) / 3;
//	}
//
//}
//
//int main(){
//
//	printf("My CUDA example.\n");
//	
//	int iWidth, iHeight, iBpp;
//	C_QPCTimer timerq_;
//	timerq_.Initialize();
//
//	double timeStart_, timeElapsed_;
//
//	vector<unsigned char> pDataIn(3840*2160*4);
//	vector<unsigned char> pDataOut;
//
//	unsigned int SizeIn, SizeOut;
//	unsigned char *PInData, *POutData, *DatIn, *DatOut;
//
//	int error1 = LoadBmpFile(L"3840x2160.bmp", iWidth, iHeight, iBpp, pDataIn);
//
//	if (error1 != 0 || pDataIn.size() == 0 || iBpp != 32)
//	{
//		printf("error load input file!\n");
//	}
//
//	//libGFL
//	pDataOut.resize(pDataIn.size()/4);	
//	//Для CUDA
//	SizeIn = pDataIn.size()/4;
//	SizeOut = pDataOut.size();
//	PInData = pDataIn.data();
//	POutData = pDataOut.data();
//	
//	//Для CPU
//	DatIn = pDataIn.data();
//	DatOut = pDataOut.data();
//	printf("CPU\n");
//	timeStart_ = timerq_.GetTime();
//	for (int j = 0; j < CYCLES; j++)
//	{
//		faktorial(SizeIn, DatIn, DatOut, iWidth, iHeight);
//	}
//	timeElapsed_ = timerq_.GetTime() - timeStart_;
//
//	cout << "Average time CPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
//
//	printf("GPU\n");
//	//my_cuda((uchar4*)PInData, POutData, SizeIn, SizeOut, iWidth, iHeight);
//
//	params par;
//	par.PInData = (uchar4*)PInData;
//	par.POutData = POutData;
//	par.SizeIn = SizeIn;
//	par.SizeOut = SizeOut;
//	par.iWidth = iWidth;
//	par.iHeight = iHeight;
//	
//	printf("Allocate memory on device\n");
//	cudaMalloc((void**)&par.devDatIn, par.SizeIn * sizeof(uchar4));
//	cudaMalloc((void**)&par.devDatOut, par.SizeOut * sizeof(unsigned char));
//
//	printf("Copy data on device\n");
//	cudaMemcpy(par.devDatIn, par.PInData, par.SizeIn * sizeof(uchar4), cudaMemcpyHostToDevice);
//	cudaMemcpy(par.devDatOut, par.POutData, par.SizeOut * sizeof(unsigned char), cudaMemcpyHostToDevice);
//
//	my_cuda(par.devDatIn, par.devDatOut);
//
//	cudaMemcpy(par.POutData, par.devDatOut, par.SizeOut * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//
//	cudaFree(par.devDatOut);
//	cudaFree(par.devDatIn);
//
//	my_cuda_func3(&par);
//
//	return 0;
//}

*/



												///////Предыдущий код///////

//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <conio.h>
//#include <string.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "test.h"
//
//using namespace std;
//void faktorial(int InSize, uchar *DataIn, uchar *DataOut)
//{
//    for (int i = 0, j = 0; i < InSize; i += 3, j++)
//    {
//        DataOut[j] = (DataIn[i] + DataIn[i + 1] + DataIn[i + 2]) / 3;
//    }
//}
//
//void my_cuda_func3(unsigned char *DatOut, int SizeOut, unsigned int iWidth, unsigned int iHeight)
//{
//	IplImage *image2 = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
//	image2->imageData = (char*)DatOut;
//	cvNamedWindow("Gray");
//	cvShowImage("Gray", image2);
//	/*C_QPCTimer timerq_;
//	timerq_.Initialize();	
//	double timeStart_, timeElapsed_;
//	int cyclesin = 100;
//	printf("Processing writing the picture data in file for %d iterations\n", CYCLES);
//	timeStart_ = timerq_.GetTime();
//	for (int j = 0; j < CYCLES; j++)
//	{
//		WriteBmpFile(L"3840x2160_gray.bmp", iWidth, iHeight, 8, SizeOut * sizeof(unsigned char), POutData, false);
//	}
//	timeElapsed_ = timerq_.GetTime() - timeStart_;
//
//	cout << "Average time GPU: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;*/
//	printf("Done.\n");
//}
////void faktorial(int InSize, char *DataIn, char *DataOut)// заголовок функции
////{
////	for(int i = 0,  j = 0; i < InSize; i += 4, j++)
////	{
////		DataOut[j] = (DataIn[i] + DataIn[i + 1] + DataIn[i + 2]) / 3;
////	}
////
////}
//
//int main() 
//{
//
//		
//        // задаём высоту и ширину картинки
//        int height = 620;
//        int width = 440;
//		char* c = "C:\Users\Sergey\Documents\Visual Studio 2012\Projects\CudamyExample\3840x2160.bmp";
//		//printf("Input data");
//		printf("Input source of image\n Example of right directory file: E:\henrik-evensen-castle-valley-v03.jpg\n Your try:\n");
//		char *tbLEN;
//		tbLEN = new char [1024];
// 
//		cin.getline(tbLEN,1024);
// 
//		cout << tbLEN;
//
//		
//		IplImage* image;
//		image = cvLoadImage(tbLEN, 1);
//		//int width = ;
//		int height1 = image->height;
//		int width1 = image->width;
//		int step = image->widthStep;
//	    int SizeIn = step*height1;
//		cout<< SizeIn/4<< "Hello"<<endl;
//		//char* DatIn = image->imageData;
//		IplImage *image2 = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 1);
//		unsigned int SizeOut = SizeIn;
//		uchar* DatIn = (uchar*)image->imageData;
//		unsigned char* DatOut = (unsigned char*)image2->imageData;
//		//char* DatOut = image2->imageData;
//
//		faktorial(SizeIn, DatIn, DatOut);
//		my_cuda((uchar4*)DatIn, /*(uchar4*)*/DatOut, SizeIn, SizeOut, height1, width1);
//		cvNamedWindow("Imagecolor");
//		cvShowImage("Imagecolor", image);
//
//		//cvNamedWindow("Gray");
//		//cvShowImage("Gray", image2);
//
//
//        cvWaitKey(0);
//        return 0;
//}

//#include "stdafx.h"
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <conio.h>
//
//void main()
//{
//  std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
//  _getch();
//}

//#include "cuda_runtime.h"
//#include <iostream>
//#include <ctime>
//#include <stdio.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "device_launch_parameters.h"
//using namespace std;
//using namespace cv;
//
//__global__ void convertImage(int width, int height, int nchannels, int step, uchar *d_data, int nchannels2, int step2, uchar *d_data2)
//{
//int i, j, r, g, b, byte, z = 0;
//for(i=0; i<height; i++)
//    for(j=0; j<width; j++)
//    {
//        r = d_data[i*step + j*nchannels + 0];
//        g = d_data[i*step + j*nchannels + 1];
//        b = d_data[i*step + j*nchannels + 2];
//
//        byte = (r+g+b)/3;
//
//        d_data2[i*step2 + j*nchannels2 + 0] = byte;
//        d_data2[i*step2 + j*nchannels2 + 1] = byte;
//        d_data2[i*step2 + j*nchannels2 + 2] = byte;
//    }
//}
//
//int main()
//{
//IplImage *img = cvLoadImage("E:\Assassin.jpg", CV_LOAD_IMAGE_COLOR);
//int width = img->width;
//int height = img->height;
//int nchannels = img->nChannels;
//int step = img->widthStep;
//cout<<"Image 1 : "<<width<<"\t"<<height<<"\t"<<nchannels<<"\t"<<step<<endl;
//uchar *data = (uchar*)img->imageData;
//uchar *d_data;
//int size = step * height;
//cudaMalloc(&d_data, size);
//cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
//
//IplImage *img2 = cvCreateImage(cvSize(img->height, img->width), IPL_DEPTH_8U, 1);
//int width2 = img2->width;
//int height2 = img2->height;
//int nchannels2 = img2->nChannels;
//int step2 = img2->widthStep;
//cout<<"Image 2 : "<<width2<<"\t"<<height2<<"\t"<<nchannels2<<"\t"<<step2<<endl;
//uchar *data2 = (uchar*)img2->imageData;
//uchar *d_data2;
//int size2 = step2 * height2;
//cudaMalloc(&d_data2, size2);
//
//long long i;
//uchar *temp = data;
//
//convertImage<<<1,1>>>(width, height, nchannels, step, d_data, nchannels2, step2, d_data2);
//cudaMemcpy(data2, d_data2, size, cudaMemcpyHostToDevice);
//
//  cudaMemcpy(data2, d_data2, size2, cudaMemcpyDeviceToHost);
//
//cvNamedWindow("Imagecolor");
//cvShowImage("Imagecolor", img);
//
//cvNamedWindow("Gray");
//cvShowImage("Gray", img2);
//
//cvWaitKey();
//
//return 0;
//}

//

