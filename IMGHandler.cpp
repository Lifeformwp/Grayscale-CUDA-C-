#include "IMGHandler.h"

	void ImgWork::setData(vector<Mat>::iterator img, vector<Mat>::iterator img3, int NumThreadX)
	{
		SizeINImg = (*img).step * (*img).rows;
		SizeOutImg = (*img3).step * (*img3).rows;
		DataImg = (*img).data;
		DataImg2 = (*img3).data;
		BlocksNumber = ((*img).cols * (*img).rows)/NumThreadX;
	}

	void ImgWork::setDataSingle(Mat img, Mat img3, int NumThreadX)
	{
		SizeINImg = img.step * img.rows;
		SizeOutImg = img3.step * img3.rows;
		DataImg = img.data;
		DataImg2 = img3.data;
		BlocksNumber = (img.cols * img.rows)/NumThreadX;
	}

	int ImgWork::getSizeBlocks()
	{
		return SizeINImg, BlocksNumber, SizeOutImg;
	}

	unsigned char ImgWork::getData12()
	{
		return *DataImg, *DataImg2;
	}


	void Grayscale::GSLightness()
	{
		for(int i = 0,  j = 0; i < SizeINImg; i += 3, j++)
		{
			int min= DataImg[i]<DataImg[i + 1]?DataImg[i]:DataImg[i + 1];
			min= DataImg[i + 2]<min?DataImg[i + 2]:min;
			int max = DataImg[i]<DataImg[i + 1]?DataImg[i]:DataImg[i + 1];
			max=DataImg[i + 2]<max?DataImg[i + 2]:max;
			DataImg2[j] = (max + min)/2;
		}
	}

	void Grayscale::GSLuminosity()
	{
		for(int i = 0,  j = 0; i < SizeINImg; i += 3, j++)
		{
			DataImg2[j] = ((0.21 * DataImg[i]) + (0.72 * DataImg[i + 1]) + (0.07 * DataImg[i + 2]));
		}
	}

	void Grayscale::GSAverage()
	{
		for(int i = 0,  j = 0; i < SizeINImg; i += 3, j++)
		{
			DataImg2[j] = (DataImg[i] + DataImg[i + 1] + DataImg[i + 2])/3;
		}
	}
