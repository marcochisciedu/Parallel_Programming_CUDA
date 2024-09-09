
#ifndef PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH
#define PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH

#include <string>
#include <random>
#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include <iostream>
struct Image {
    int * pixels;
    int width ;
    int height ;
};

Image getImageFromFile(const std::string& filename);

void saveImageToFile(Image image, const std::string& filename,  const std::string& directory);

bool compareImages(Image first_img, Image second_img);

Image generateImage(int width, int height, int seed);

#endif //PARALLEL_PROGRAMMING_CUDA_IMAGE_CUH
