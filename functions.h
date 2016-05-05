#pragma once

class BlackWhiteImg
{

public:

	void setData(int InSize, unsigned char *DataOut, unsigned char *DataIn);


	void Grayscale();

	unsigned char getGrayscaleData();

private:
	unsigned char *DataInClass, *DataOutClass;
	int SizeInClass;
};
