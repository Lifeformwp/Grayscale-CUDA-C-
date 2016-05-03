# Grayscale-CUDA-C-

This is my hobby project in which I'm trying to learn new stuff about concurrent computing. Concurrent computing is a form of computing in which several computations are executing during overlapping time periods.

This project will have two versions, each one for it's own way.

In first version (on which I'm working right now) of my programm I will use CUDA, OpenCL and C++ threads to show the difference between them and see which one will show the best result. Also I'm using OpenCV to load and work with different images, though it hard to work with Alpha channel in this framework. But OpenCV has a big advantage, which I will use a bit later - computer vision. Computer vision is a field that includes methods for acquiring, processing, analyzing, and understanding images and, in general, high-dimensional data from the real world in order to produce numerical or symbolic information, e.g., in the forms of decisions.

So in the end of second version (on which I'll work after the first would be done) of my programm I will get the data through the webcam using OpenCV, operate with it (make it black and white or anything else) using CUDA, OpenCL and C++ threads, and then save the output to some folder. This will truly show the potential of concurrent computing using different tools and maybe someone will use it for it's own sake.

For now I have the following results:
1) Done loading/showing/saving images using up to date OpenCV C++ functional;
2) Wrapper for working with CUDA;
3) CUDA Malloc, Memcpy, "black-and-white" kernel, which processing image and give the results back to main.cpp;
4) CPU part in main.cpp, just a little function to work with process the image on CPU;
5) OpenCL programm (separate, I will add it combine it with this project very soon), which processing image on different GPU/CPU;
6) Errorcheck for CUDA functions.

Soon: CUDA streams functionality, OpenCL, classes.
