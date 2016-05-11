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
#include "IMGHandler.h"


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
	C_QPCTimer timerq_;
	timerq_.Initialize();
	double timeStart_, timeElapsed_;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	ImgWork *ImageWork = new ImgWork;
	Grayscale *GS = new Grayscale;
		cudaStream_t* streames = new cudaStream_t[15];
	for (int i = 0; i < 5;i++) {
                cudaStreamCreate(&streames[i]);
	}
	
	int NumThreadX = deviceProp.maxThreadsPerBlock;
 Mat input;
 //Mat frame;
VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
char strfps[64] = "0 fps";
char strsec[64] = "0 sec";
if (!stream1.isOpened()) { //check if video device has been initialised
cout << "cannot open camera";
}
clock_t start = clock();
unsigned int fps = 0;
//unconditional loop
params par;
double time = 0;
int WebCamNum = 0;
bool rpt = true;
//GPU WEB CAM PROCESSING
	cout <<"Input number of algorithm for GPU converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
	cin >>WebCamNum;
	switch(WebCamNum)
	{
	case 1:
		
		while (true) {
	stream1.read(input);
Mat frame(input.rows, input.cols, CV_8UC1);
	ImageWork->setDataSingle(input, frame, NumThreadX);

		
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));

		timeStart_ = timerq_.GetTime();
		gpuErrchk(cudaMemcpy(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		
my_cuda1(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);

		gpuErrchk(cudaMemcpy(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
			cout << "Average time GPU Luminosity: " << time << " ms" << endl;
			sprintf (strsec, "Average time %f ms", time);
		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);
		fps++;
	if(clock() - start > CLOCKS_PER_SEC)
	{
		sprintf(strfps, "GPU Luminosity %d fps", fps);
		fps = 0;
		start = clock();
	}

	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
	if (waitKey(30) >= 0)
	{
	rpt = false;
	destroyWindow("cam");
	break;
	}
	}
	rpt = false;	
	break;
	case 2:
		while (true) {
	stream1.read(input);
Mat frame(input.rows, input.cols, CV_8UC1);
	ImageWork->setDataSingle(input, frame, NumThreadX);

		
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));

		timeStart_ = timerq_.GetTime();
		gpuErrchk(cudaMemcpy(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		
my_cuda2(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);

		gpuErrchk(cudaMemcpy(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
			cout << "Average time GPU Lightness: " << time << " ms" << endl;
			sprintf (strsec, "Average time %f ms", time);
		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);
		fps++;
	if(clock() - start > CLOCKS_PER_SEC)
	{
		sprintf(strfps, "GPU Lightness %d fps", fps);
		fps = 0;
		start = clock();
	}

	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
	if (waitKey(30) >= 0)
	{
	rpt = false;
	destroyWindow("cam");
	break;
	}
	}
	break;
	rpt = false;
	case 3:
		while (true) {
	stream1.read(input);
Mat frame(input.rows, input.cols, CV_8UC1);
	ImageWork->setDataSingle(input, frame, NumThreadX);

		
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));

		timeStart_ = timerq_.GetTime();
		gpuErrchk(cudaMemcpy(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		
my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);

		gpuErrchk(cudaMemcpy(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
			cout << "Average time GPU Average: " << time << " ms" << endl;
			sprintf (strsec, "Average time %f ms", time);
		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);
		fps++;
	if(clock() - start > CLOCKS_PER_SEC)
	{
		sprintf(strfps, "GPU Average %d fps", fps);
		fps = 0;
		start = clock();
	}

	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
	if (waitKey(30) >= 0)
	{
	rpt = false;
	destroyWindow("cam");
	break;
	}
		rpt = false;
		}
		break;
		 default:
		cout<<"Incorrect input"<<endl;
		rpt = true;
		break;
	}


//CPU WEB CAM PROCESSING
		cout <<"Input number of algorithm for CPU converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
	cin >>WebCamNum;
	switch(WebCamNum)
	{
	case 1:
			while (true) {
	stream1.read(input);
	Mat frame(input.rows, input.cols, CV_8UC1);
	GS->setDataSingle(input, frame, NumThreadX);
	timeStart_ = timerq_.GetTime();
	GS->GSLuminosity();
	timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
				cout << "Average time CPU Luminosity: " << time << " ms" << endl;
				sprintf (strsec, "Average time %f ms", time);

	fps++;
		if(clock() - start > CLOCKS_PER_SEC)
		{
			sprintf(strfps, "%d fps", fps);
			fps = 0;
			start = clock();
		}
	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
		if (waitKey(30) >= 0)
		{
		rpt = false;
		destroyWindow("cam");
		break;
		}
	}
				break;

		case 2:
			while (true) {
	stream1.read(input);
	Mat frame(input.rows, input.cols, CV_8UC1);
	GS->setDataSingle(input, frame, NumThreadX);
	timeStart_ = timerq_.GetTime();
	GS->GSLightness();
	timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
				cout << "Average time CPU Lightness: " << time << " ms" << endl;
				sprintf (strsec, "Average time %f ms", time);
	fps++;
		if(clock() - start > CLOCKS_PER_SEC)
		{
			sprintf(strfps, "%d fps", fps);
			fps = 0;
			start = clock();
		}
	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
		if (waitKey(30) >= 0)
		{
		rpt = false;
		destroyWindow("cam");
		break;
		}
	}
			break;
		case 3:
			while (true) {
	stream1.read(input);
	Mat frame(input.rows, input.cols, CV_8UC1);
	GS->setDataSingle(input, frame, NumThreadX);
	timeStart_ = timerq_.GetTime();
	GS->GSAverage();
	timeElapsed_ = timerq_.GetTime() - timeStart_;
		time = (timeElapsed_ / (double)1);
				cout << "Average time CPU Average: " << time << " ms" << endl;
				sprintf (strsec, "Average time %f ms", time);
	fps++;
		if(clock() - start > CLOCKS_PER_SEC)
		{
			sprintf(strfps, "%d fps", fps);
			fps = 0;
			start = clock();
		}
	putText(frame, strsec, cv::Point(0,20), 2, 0.8,(0,0, 0));
	putText(frame, strfps, cv::Point(0,40), 2, 0.8,(0,0, 0));
	imshow("cam", frame);
		if (waitKey(30) >= 0)
		{
		rpt = false;
		destroyWindow("cam");
		break;
		}
	}
			break;
		}

//while (true) {
//	stream1.read(input);
//	Mat frame(input.rows, input.cols, CV_8UC1);
//	ImageWork->setDataSingle(input, frame, NumThreadX);
//	par.chard = (unsigned char*)streames[0];
//
//		timeStart_ = timerq_.GetTime();
//		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
//		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));
//
//		
//		cudaMemcpyAsync(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice,streames[1]);
//		cudaStreamSynchronize(streames[1]);
//		cudaMemcpyAsync(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice, streames[2]);
//		cudaStreamSynchronize(streames[2]);
//		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
//		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
//		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
//		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
//		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
//		cudaStreamSynchronize(streames[0]);
//
//		cudaMemcpyAsync(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost, streames[3]);
//		cudaStreamSynchronize(streames[3]);
//
//		timeElapsed_ = timerq_.GetTime() - timeStart_;
//			cout << "Average time CPU Average: " << (timeElapsed_ / (double)1) << " ms" << endl;
//		cudaFree(par.devDatOut);
//		cudaFree(par.devDatIn);
//		for (int m; m < 5; m++) {
//                cudaStreamDestroy(streames[m]);
//	}
//		fps++;
//	if(clock() - start > CLOCKS_PER_SEC)
//	{
//		sprintf(strfps, "%d fps", fps);
//		fps = 0;
//		start = clock();
//	}
//	putText(frame, strfps, cv::Point(0,20), 3, 1, CV_RGB(25,200,25));
//	imshow("cam", frame);
//	if (waitKey(30) >= 0)
//break;
//}

	
	/*double timeStart_, timeElapsed_;*/
	int i = 0, k = 0, m = 0;
	int AlgorNum = 0, GPUnum = 0;
	bool repeat = false;
	bool returned = false;


	
	string str;
	getline(cin, str);
	//C_QPCTimer timerq_;
	//timerq_.Initialize();
	int ImageAmount = 0;
	vector<Mat> images; 
	vector<Mat> imagesgrey; 
	vector<Mat> imagesgreycpu;
	vector<cudaStream_t> Streams;

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
	destroyWindow("Vector of imgs");
	vector<Mat>::iterator itcpu;
	vector<Mat>::iterator itcpu2;
do{
	cout <<"Input number of algorithm for CPU converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
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
		destroyWindow("CPU Grayscale Luminosity");
		break;
		
	case 2:
		for(itcpu=images.begin(), itcpu2=imagesgreycpu.begin(); itcpu!=images.end(),itcpu2!=imagesgreycpu.end(); itcpu++,itcpu2++)
		{
			GS->setData(itcpu, itcpu2, NumThreadX);
			timeStart_ = timerq_.GetTime();
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
		destroyWindow("CPU Grayscale Lightness");
		break;

	case 3:
		for(itcpu=images.begin(), itcpu2=imagesgreycpu.begin();itcpu!=images.end(),itcpu2!=imagesgreycpu.end();itcpu++,itcpu2++)
		{
			GS->setData(itcpu, itcpu2, NumThreadX);
			timeStart_ = timerq_.GetTime();
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
		destroyWindow("CPU Grayscale Average");
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


	/*params par;*/
	cout <<"Input number of algorithm for GPU converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
	cin >>GPUnum;
	for (k, it=images.begin(), it2=imagesgrey.begin();k<ImageAmount,it!=images.end(),it2!=imagesgrey.end();k++, it++, it2++) {
		ImageWork->setData(it, it2, NumThreadX);
		par.chard = (unsigned char*)stream[k];
		
		
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));
	
		gpuErrchk(cudaMemcpyAsync(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice,stream[k+1]));
		gpuErrchk(cudaStreamSynchronize(stream[k+1]));
		gpuErrchk(cudaMemcpyAsync(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[k+2]));
		gpuErrchk(cudaStreamSynchronize(stream[k+2]));


do{
	switch(GPUnum)
	{
	case 1:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda1(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
		cudaStreamSynchronize(stream[k]);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Luminosity: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		GPUnum = 1;
		break;

	case 2:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda2(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
		cudaStreamSynchronize(stream[k]);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Lightness: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		GPUnum = 2;
		break;

	case 3:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, par.chard);
		cudaStreamSynchronize(stream[k]);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Average: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		break;
    default:
		cout<<"Incorrect input"<<endl;
		repeat = true;
		break;
	}
		}while (returned == true);

		gpuErrchk(cudaMemcpyAsync(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[k+3]));
		gpuErrchk(cudaStreamSynchronize(stream[k+3]));
	}

	for (it=images.begin(), it2=imagesgrey.begin();i<ImageAmount,it!=images.end(),it2!=imagesgrey.end();it++, it2++) 
	{
                
			imshow("myWinMemcpy", (*it));
			 waitKey(1000);
			  imshow("myWinMemcpy", (*it2));
			 waitKey(1000);
	}
	destroyWindow("myWinMemcpy");


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
		Mat imgsinglecpu(imgsingle.rows,imgsingle.cols, CV_8UC1);
		ImageWork->setDataSingle(imgsingle, imgsingle2, NumThreadX);

		printf("Allocating memory on Device\n");
		gpuErrchk(cudaMalloc((void**)&par.devDatIn, ImageWork->SizeINImg * sizeof(unsigned char)));
		gpuErrchk(cudaMalloc((void**)&par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char)));

		printf("Copy data on Device\n");
		gpuErrchk(cudaMemcpy(par.devDatIn, ImageWork->DataImg, ImageWork->SizeINImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(par.devDatOut, ImageWork->DataImg2, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyHostToDevice));
		

		do{
	cout <<"Input number of algorithm for converting color to grayscale" <<endl << "1 - Luminosity" <<endl <<"2 - Lightness" <<endl<< "3 - Average"<<endl;
	cin >>AlgorNum;
	do{
	
	switch(AlgorNum)
	{
	case 1:
			GS->setDataSingle(imgsingle, imgsinglecpu, NumThreadX);
			timeStart_ = timerq_.GetTime();
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSLuminosity();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time SingleCPU Luminosity: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
			imshow("CPU Grayscale Luminosity",(imgsinglecpu));
			waitKey(0);
			repeat = false;
			destroyWindow("CPU Grayscale Luminosity");
		break;
		
	case 2:	
			GS->setDataSingle(imgsingle, imgsinglecpu, NumThreadX);
			timeStart_ = timerq_.GetTime();
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSLightness();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time SingleCPU Lightness: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;			
			imshow("CPU Grayscale Lightness",(imgsinglecpu));
			waitKey(0);
			repeat = false;
			destroyWindow("CPU Grayscale Lightness");
		break;

	case 3:
		
			GS->setDataSingle(imgsingle, imgsinglecpu, NumThreadX);
			timeStart_ = timerq_.GetTime();
			for (int k = 0; k<CYCLES;k++)
			{
			GS->GSAverage();
			}
			timeElapsed_ = timerq_.GetTime() - timeStart_;
			cout << "Average time SingleCPU Lightness: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;				
			imshow("CPU Grayscale Average",(imgsinglecpu));
			waitKey(0);
			repeat = false;
			destroyWindow("CPU Grayscale Average");
		break;
	default:
		cout<<"Incorrect input"<<endl;
		repeat = true;
		break;
	}
	}while(repeat == true);


	switch(AlgorNum)
	{
	case 1:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda1(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Luminosity single image: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		GPUnum = 1;
		break;

	case 2:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda2(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Lightness single image: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		GPUnum = 2;
		break;

	case 3:
		timeStart_ = timerq_.GetTime();
		for (int a = 0; a < CYCLES; a++)
		{
		my_cuda3(par.devDatIn, par.devDatOut, ImageWork->BlocksNumber, NumThreadX, 0);
		}
		timeElapsed_ = timerq_.GetTime() - timeStart_;
		cout << "Average time GPU Average single image: " << (timeElapsed_ / (double)CYCLES) << " ms" << endl;
		returned = false;
		break;
    default:
		cout<<"Incorrect input"<<endl;
		repeat = true;
		break;
	}
		}while (returned == true);
	
		gpuErrchk(cudaMemcpy(ImageWork->DataImg2, par.devDatOut, ImageWork->SizeOutImg * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		cudaFree(par.devDatOut);
		cudaFree(par.devDatIn);

		namedWindow("Color image", CV_WINDOW_AUTOSIZE);
		imshow("Color image", imgsingle);

		namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
		imshow("Gray image", imgsingle2);
		
		waitKey(0);
		destroyAllWindows();
		delete ImageWork;
		delete GS;
        return 0;
}
