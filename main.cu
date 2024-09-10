#define MAX_KERNEL_DIM (10*10)
__constant__ float kernel_pixels[MAX_KERNEL_DIM];

#include "Image.cuh"
#include "SequentialConvolution.cuh"
#include "GPUConvolution.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#include <chrono>
__host__ double timeSequential(Image input, Kernel kernel, Image output){

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    sequential2DConvolution(input.pixels, kernel.pixels, output.pixels,
                            input.width, input.height,kernel.size );
    auto t2 = Clock::now();
    std::chrono::duration<double, std::milli> time = t2 - t1;

    return time.count();
}

__host__ double timeGPU(Image input, Kernel kernel, Image output, dim3 grid, dim3 block ){
    Image dev_input{}, dev_output{};
    dev_input = allocateOnDevice(input);
    dev_output = allocateOnDevice(output);

    // move pixel value to constant memory
    cudaMemcpyToSymbol(kernel_pixels, kernel.pixels, kernel.size*kernel.size*sizeof(float),
                       0, cudaMemcpyHostToDevice );

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    GPU2DConvolution<<<grid, block>>>(dev_input.pixels, dev_output.pixels,
                                      dev_input.width, dev_input.height,kernel.size );
    cudaDeviceSynchronize();

    auto t2 = Clock::now();
    std::chrono::duration<double, std::milli> time = t2 - t1;

    freeImageDev(dev_input);
    copyFromDeviceToHost(dev_output, output);
    freeImageDev(dev_output);

    return time.count();
}

int main() {
    //Image gen_img = generateImage(512, 512, 11);
    //saveImageToFile(gen_img, "rand.png", "Grayscale images");

    Image image = getImageFromFile("cat.png");
    Image output_image_seq = copyImage(image);
    Image output_image_GPU = copyImage(image);
    Kernel kernel = generateBlurKernel(5);

    double time_seq = timeSequential(image, kernel, output_image_seq);
    printf("The sequential 2D convolution completed in %fms \n", time_seq);
    saveImageToFile(output_image_seq, "blur_cat.png", "Output grayscale images");

    // BLOCK_WIDTH is defined for kernel size = 5
    int out_tile_width =  12;
    int block_width = (out_tile_width + kernel.size -1) ;
    dim3 grid((image.width-1)/out_tile_width+1, (image.height-1)/out_tile_width+1,1);
    dim3 block(block_width, block_width);

    double time_GPU = timeGPU(image, kernel, output_image_GPU, grid, block);
    double speedup_GPU = time_seq/time_GPU;
    printf("The GPU 2D convolution completed in %fms with a speedup of %f \n", time_GPU, speedup_GPU);
    saveImageToFile(output_image_GPU, "blur_cat_GPU.png", "Output grayscale images");

    compareImages(output_image_seq, output_image_GPU);


    freeImageHost(image);
    freeImageHost(output_image_seq);
    freeImageHost(output_image_GPU);

    return 0;
}
