
#ifndef PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH
#define PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH

#include <string>
#include <random>
#include <cassert>
#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include <iostream>
struct Image {
    int * pixels;
    int width ;
    int height ;
};

struct Kernel{
    float* pixels;
    int size;
};

__host__ Image getImageFromFile(const std::string& filename);

__host__ void saveImageToFile(Image image, const std::string& filename,  const std::string& directory);

__host__ void compareImages(Image first_img, Image second_img);

__host__ Image generateImage(int width, int height, int seed);

__host__ Kernel generateBlurKernel(int size);

__host__ Image copyImage(Image const &image);

__host__ void copyFromHostToDevice(const Image& hostImage, Image& devImage);

__host__ Image allocateOnDevice(Image const &hostImage);

__host__ void copyFromDeviceToHost(Image const &devImage, Image &hostImage);

__host__ void freeImageHost(Image &hostImage);

__host__ void freeImageDev(Image &devImage);


#endif //PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH
