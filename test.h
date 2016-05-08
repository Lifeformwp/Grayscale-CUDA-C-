struct params
{
	unsigned char *chard;
	unsigned char *devDatIn;
	unsigned char *devDatOut;
};

extern "C" int my_cuda1(unsigned char *devDatIn, unsigned char *devDatOut, int NumBlocksX, int NumThreadsX, unsigned char* chard);
