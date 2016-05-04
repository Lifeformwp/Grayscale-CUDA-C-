# Grayscale-CUDA-C-

This is my hobby project in which I'm trying to learn new stuff about concurrent computing. Concurrent computing is a form of computing in which several computations are executing during overlapping time periods.

This project will have two versions, each of them in it own way.

In the first version (which I'm working on right now) of my programm I will use CUDA, OpenCL and C++ threads to show the difference between them and see which one will show the best result. Also I'm using OpenCV to load and work with different images, though it's hard to work with Alpha channel in this framework. But OpenCV has a big advantage, which I will use a bit later - computer vision. Computer vision is a field that includes methods for acquiring, processing, analyzing, and understanding images and, in general, high-dimensional data from the real world in order to produce numerical or symbolic information, e.g., in the forms of decisions.

So in the end of the second version (on which I'll work after the first one is done) of my programm I will get the data through the webcam using OpenCV, transform it (make it black and white or do anything else) using CUDA, OpenCL and C++ threads, and then save the output in a folder. This will truly show the potential of concurrent computing by using different tools and maybe someone will use it for his own sake.

For now I have the following results:
1) loading/showing/saving images using up to date OpenCV C++ functional;
2) A wrapper for working with CUDA;
3) CUDA Malloc, Memcpy, "black-and-white" kernel, which processes images and gives the results back to the main.cpp;
4) CPU part in main.cpp, just a small function which processes images on CPU;
5) OpenCL programm (separately, I will combine it with this project very soon), which processes images on different GPU/CPU;
6) Errorcheck for CUDA functions.

Soon: CUDA streams functionality, OpenCL, classes.
