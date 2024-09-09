#include "Image.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"


int main() {
    Image image = getImageFromFile("cat.png");
    saveImageToFile(image,"copy_cat.png", "Output grayscale images");
    Image copy_image = getImageFromFile("copy_cat.png");
    Image copy_image2 = getImageFromFile("copy_cat2.png");
    std::cout<< compareImages(copy_image2, copy_image) << std::endl;

    Image gen_img = generateImage(512, 512, 11);
    saveImageToFile(gen_img, "rand.png", "Grayscale images");

    return 0;
}
