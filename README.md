# Parallel Programming Projects CUDA

## Report
The report that details the parallelization of a sequential 2D
convolution algorithm using CUDA is in the Report folder.

## Used data
The stb_image contains the file used to read the real input images.

A basic grayscale image of a lynx was used to visualize the results of the 2D convolution. The images are contained in Grayscale images and Output grayscale images. 

To test the performances of the different 2D convolution implementations, randomly generated images of varying sizes were utilized.
The Image structure is defined and implemented in Image.cu/cuh alongside its functions.

The same files also define and implement the Kernel structure. When all the weights in the kernel are set to 1/(kernel
size * kernel size), the convolution functions as a box blur.

## 2D Convolution
SequentialConvolution.cu/cuh: provides a simple sequential implementation of 2D convolution. Two for loops iterate over every
pixel of the input image and calculate the
intensity of the output pixel given the weights of the kernel.
 
GPUConvolution.cu/cuh: contains two CUDA-based versions of 2D convolution (both with or without the kernel in the constant memory). GPU2DConvolution is a simple CUDA-based 2D
convolution while GPU2DConvolutionTiling is a 2D CUDA-based convolution with tiling.

main.cu: the executable file that tests and compares the different convolution implementations using various image sizes, CUDA blocks, and kernels. 
The sequential and CUDA 2D convolutions are run multiple times, calculating the average execution
times and speedups.

The results are saved in a txt file and are found in the results folder.

If the CUDA-based 2D convolution produces the same
blurred output as the sequential version when using the
same kernel, it is considered correct.
