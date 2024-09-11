
#include "Image.cuh"
// create grayscale Image from file
__host__ Image getImageFromFile(const std::string& filename){
    // load image, no transparency
    int width, height, original_no_channels;
    int channels = 3;
    unsigned char *img = stbi_load(("Grayscale images/"+filename).c_str(), &width, &height,
                                   &original_no_channels, channels);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    // create grayscale Image
    Image out_image{};
    out_image.width = width;
    out_image.height = height;
    out_image.pixels = (int *) malloc(out_image.width * out_image.height * sizeof(int));

    // Convert the input image to gray and save it
    for(int i = 0; i < out_image.height; i++)
    {
        for(int j = 0; j < out_image.width; j++)
        {
            out_image.pixels[ i * out_image.width + j] = (int)((img[ channels*(i * out_image.width + j)]+
                                                               img[ channels*(i * out_image.width + j )+1]+
                                                               img[ channels*(i * out_image.width + j) +2])/3);
        }
    }

    return out_image;
}

// save the grayscale Image to file
__host__ void saveImageToFile(Image image, const std::string& out_filename, const std::string& directory) {

    int img_size = image.width * image.height;
    auto *img = static_cast<unsigned char *>(malloc(img_size));
    if(img == NULL) {
        printf("Unable to allocate memory for the image.\n");
        exit(1);
    }

    // convert the pixels
    for(int i = 0; i < image.height; i++)
    {
        for(int j = 0; j < image.width; j++)
        {
            img[ i * image.width + j] = (unsigned char)image.pixels[ i * image.width + j];
        }
    }

    stbi_write_jpg((directory+"/"+out_filename).c_str(), image.width, image.height,
                   1, img, 100);
}

// compare two output Images to check the results
__host__ void compareImages(Image first_img, Image second_img, std::ofstream& outfile){
    if(first_img.height != second_img.height){
        printf("The images have different heights \n");
        outfile<< "The images have different heights" <<std::endl;
        exit(1);
    }
    if(first_img.width != second_img.width){
        printf("The images have different widths \n");
        outfile<< "The images have different widths" <<std::endl;
        exit(1);
    }
    for(int i = 0; i < first_img.width * first_img.height; i++){
            if(first_img.pixels[ i ] != second_img.pixels[ i ]){
                printf("The images are different at pixel %d \n", i+1);
                outfile<< "The images are different at pixel" << i+1<<std::endl;
                printf("%d \n", first_img.pixels[i]);
                printf("%d \n", second_img.pixels[i]);
                exit(1);
            }
    }
    printf("The images are the same \n");
    outfile<< "The images are the same" <<std::endl;
}

// generate fake Image given its size
__host__ Image generateImage(int width, int height, int seed) {
    // create image
    Image image{};
    image.width = width;
    image.height = height;
    image.pixels = (int *) malloc(image.width * image.height * sizeof(int));

    // random integer generation
    std::random_device rd;
    std::default_random_engine eng(rd());
    eng.seed(seed);
    std::uniform_int_distribution<> dis(0, 1000);

    // fill image with random pixels
    for(int i = 0; i < image.width * image.height; i++){
        image.pixels[i] = dis(eng);
    }

    return image;
}

// generate square Blur Kernel given its size
__host__ Kernel generateBlurKernel(int size){
    Kernel kernel{};
    kernel.size= size;
    kernel.pixels = (float*) malloc(size * size * sizeof(float));
    for(int i=0; i<size*size; i++){
        kernel.pixels[i]= 1/(float)(size*size);
    }
    return kernel;
}

// create a copy of the given Image
__host__ Image copyImage(const Image& image) {
    Image copy{};
    copy.width = image.width;
    copy.height = image.height;
    copy.pixels = (int*) malloc(image.width * image.height * sizeof(int));

    for(int i = 0; i < copy.width * copy.height; i++)
        copy.pixels[i] = image.pixels[i];

    return copy;
}

__host__ void copyFromHostToDevice(const Image& hostImage, Image& devImage) {
    assert(hostImage.height == devImage.height && hostImage.width == devImage.width);
    cudaMemcpy(devImage.pixels, hostImage.pixels, devImage.width*devImage.height* sizeof(int),
               cudaMemcpyHostToDevice);
}

// allocate Image on the device
__host__ Image allocateOnDevice(Image const &hostImage) {
    Image devImage{};
    devImage.width = hostImage.width;
    devImage.height = hostImage.height;
    cudaMalloc((void**)&devImage.pixels, devImage.width*devImage.height* sizeof(int));
    copyFromHostToDevice(hostImage, devImage);
    return devImage;
}

__host__ void copyFromDeviceToHost(Image const &devImage, Image &hostImage) {
    assert(hostImage.height == devImage.height && hostImage.width == devImage.width);
    cudaMemcpy(hostImage.pixels, devImage.pixels, devImage.width*devImage.height* sizeof(int),
               cudaMemcpyDeviceToHost);
}

__host__ void freeImageHost(Image &hostImage) {
    assert(hostImage.pixels != nullptr);
    hostImage.width = 0;
    hostImage.height = 0;
    free(hostImage.pixels);
}

__host__ void freeImageDev(Image &devImage) {
    assert(devImage.pixels != nullptr);
    devImage.width = 0;
    devImage.height = 0;
    cudaFree(devImage.pixels);
}