
#include "Image.cuh"

Image getImageFromFile(const std::string& filename){
    // load image, no transparency
    int width, height, original_no_channels;
    int channels = 3;
    unsigned char *img = stbi_load(("Grayscale images/"+filename).c_str(), &width, &height,
                                   &original_no_channels, channels);
    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    // create Image
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

void saveImageToFile(Image image, const std::string& out_filename, const std::string& directory) {

    int img_size = image.width * image.height;
    auto *img = static_cast<unsigned char *>(malloc(img_size));
    if(img == NULL) {
        printf("Unable to allocate memory for the image.\n");
        exit(1);
    }

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

bool compareImages(Image first_img, Image second_img){
    if(first_img.height != second_img.height){
        return false;
    }
    if(first_img.width != second_img.width){
        return false;
    }
    for(int i = 0; i < first_img.width * first_img.height; i++){
            if(first_img.pixels[ i ] != second_img.pixels[ i ]){
                return false;
            }
    }
    return true;
}

Image generateImage(int width, int height, int seed) {
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