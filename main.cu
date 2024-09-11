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

__host__ double timeGPU_tiling(Image input, Kernel kernel, Image output, dim3 grid, dim3 block){
    Image dev_input{}, dev_output{};
    dev_input = allocateOnDevice(input);
    dev_output = allocateOnDevice(output);

    // move pixel value to constant memory
    cudaMemcpyToSymbol(kernel_pixels, kernel.pixels, kernel.size*kernel.size*sizeof(float),
                       0, cudaMemcpyHostToDevice );

    size_t shared_memory_size = (block.x+ kernel.size -1) *  (block.x+ kernel.size -1) *sizeof(int);

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    GPU2DConvolutionTiling<<<grid, block, shared_memory_size>>>(dev_input.pixels,
                        dev_output.pixels,dev_input.width, dev_input.height,kernel.size );
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
    Image output_image_GPU_tiling = copyImage(image);
    Kernel kernel = generateBlurKernel(5);

    int block_width = 16;

    dim3 grid((image.width+ block_width -1 )/ block_width, (image.height+ block_width -1)/block_width,1);
    dim3 block(block_width, block_width);

    // outputfile to store the results
    std::ofstream outfile;
    outfile.open("results/Convolution_results.txt", std::ios_base::app);
    if (!outfile.is_open()){
        printf("Error: Unable to write to Convolution_results.txt \n");
        exit(1);
    }

    std::cout<< "Blurring "<< "cat.png" << " " << "with a "<< kernel.size<<"x"<<kernel.size<<
    " kernel"<< std::endl;
    outfile << "Blurring "<< "cat.png" << " " << "with a "<< kernel.size<<"x"<<kernel.size<<
            " kernel"<< std::endl;

    std::cout<< "The size of the square block is: "<< block_width << " and the grid contains " << grid.x<< "x"<<
    grid.y << "x" << grid.z << " blocks" << std::endl;
    outfile << "The size of the square block is: "<< block_width << " and the grid contains " << grid.x<< "x"<<
            grid.y << "x" << grid.z << " blocks" << std::endl;

    double time_seq = timeSequential(image, kernel, output_image_seq);
    printf("The sequential 2D convolution completed in %fms \n", time_seq);
    outfile << "The sequential 2D convolution completed in "<< time_seq << "ms"<< std::endl;
    saveImageToFile(output_image_seq, "blur_cat.png", "Output grayscale images");

    double time_GPU = timeGPU(image, kernel, output_image_GPU, grid, block);
    double speedup_GPU = time_seq/time_GPU;
    printf("The GPU 2D convolution completed in %fms with a speedup of %f \n", time_GPU, speedup_GPU);
    outfile << "The GPU 2D convolution completed in "<< time_GPU << "ms with a speedup of "<< speedup_GPU<< std::endl;
    saveImageToFile(output_image_GPU, "blur_cat_GPU.png", "Output grayscale images");

    compareImages(output_image_seq, output_image_GPU, outfile);

    double time_GPU_tiling = timeGPU_tiling(image, kernel, output_image_GPU_tiling, grid, block);
    double speedup_GPU_tiling = time_seq/time_GPU_tiling;
    printf("The GPU 2D convolution with tiling completed in %fms with a speedup of %f \n", time_GPU_tiling, speedup_GPU_tiling);
    outfile << "The GPU 2D convolution with tiling completed in "<< time_GPU_tiling
    << "ms with a speedup of "<< speedup_GPU_tiling<< std::endl;
    saveImageToFile(output_image_GPU_tiling, "blur_cat_GPU_tiling.png", "Output grayscale images");

    compareImages(output_image_seq, output_image_GPU_tiling, outfile);

    outfile<< std::endl;


    freeImageHost(image);
    freeImageHost(output_image_seq);
    freeImageHost(output_image_GPU);

    return 0;
}
