#include "functions.h"

	void BlackWhiteImg::setData(int InSize, unsigned char *DataOut, unsigned char *DataIn)
	{
		SizeInClass = InSize;
		DataInClass = DataIn;
		DataOutClass = DataOut;
	}


	void BlackWhiteImg::Grayscale()
	{
		for(int i = 0,  j = 0; i < SizeInClass; i += 3, j++)
		{
			DataOutClass[j] = (DataInClass[i] + DataInClass[i + 1] + DataInClass[i + 2]) / 3;
		}

	}

	unsigned char BlackWhiteImg::getGrayscaleData()
	{
		return *DataOutClass;
	}
