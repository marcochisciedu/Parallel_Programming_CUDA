
#ifndef PARALLEL_PROGRAMMING_CUDA_SEQUENTIALCONVOLUTION_CUH
#define PARALLEL_PROGRAMMING_CUDA_SEQUENTIALCONVOLUTION_CUH
#include <cmath>

__host__ void sequential2DConvolution(const int* input_img, const float* kernel, int* output_img,
                                      int width, int height, int kernel_size);


#endif //PARALLEL_PROGRAMMING_CUDA_SEQUENTIALCONVOLUTION_CUH
