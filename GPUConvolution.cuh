

#ifndef PARALLEL_PROGRAMMING_CUDA_GPUCONVOLUTION_CUH
#define PARALLEL_PROGRAMMING_CUDA_GPUCONVOLUTION_CUH
#include <cmath>

__global__ void GPU2DConvolution(const int* input_img, int* output_img,
                                 int width, int height, int kernel_size);

__global__ void GPU2DConvolutionTiling(const int* input_img,  int* output_img,
                                       int width, int height, int kernel_size);

#endif //PARALLEL_PROGRAMMING_CUDA_GPUCONVOLUTION_CUH
