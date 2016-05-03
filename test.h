struct params
{
	unsigned char *DatIn;
	unsigned char *DatOut;
	//unsigned int SizeOut;
	int SizeIn;
	int NumBlocksX;
	int NumThreadsX;
	int SizeOut;

	unsigned char  *devDatIn;
	unsigned char *devDatOut;
};

extern "C" int my_cuda1(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX);
