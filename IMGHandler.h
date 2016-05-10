#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
using namespace cv;
using namespace std;

class ImgWork
{
public:
	int SizeINImg, SizeINImg2;
	int BlocksNumber, BlocksNumber2;
	int SizeOutImg, SizeOutImg2;
	unsigned char* DataImg;
	unsigned char* DataImg2;	

	void setData(vector<Mat>::iterator img, vector<Mat>::iterator img3, int NumThreadX);
	void setDataSingle(Mat img, Mat img3, int NumThreadX);

	int getSizeBlocks();

	unsigned char getData12();
	
private:

};

class Grayscale : public ImgWork
{
public:

	void GSLightness();

	void GSLuminosity();

	void GSAverage();

};



