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

__host__ double timeGPU_kernel(Image input, Kernel kernel, Image output, dim3 grid, dim3 block ){
    Image dev_input{}, dev_output{};
    dev_input = allocateOnDevice(input);
    dev_output = allocateOnDevice(output);

    Kernel dev_kernel{};
    dev_kernel= allocateKernelOnDevice(kernel);

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    GPU2DConvolution_kernel<<<grid, block>>>(dev_input.pixels, dev_output.pixels, dev_kernel.pixels,
                                      dev_input.width, dev_input.height,kernel.size );
    cudaDeviceSynchronize();

    auto t2 = Clock::now();
    std::chrono::duration<double, std::milli> time = t2 - t1;

    freeImageDev(dev_input);
    copyFromDeviceToHost(dev_output, output);
    freeImageDev(dev_output);

    freeKernelDev(dev_kernel);

    return time.count();
}

__host__ double timeGPU_tiling_kernel(Image input, Kernel kernel, Image output, dim3 grid, dim3 block){
    Image dev_input{}, dev_output{};
    dev_input = allocateOnDevice(input);
    dev_output = allocateOnDevice(output);

    Kernel dev_kernel{};
    dev_kernel= allocateKernelOnDevice(kernel);


    size_t shared_memory_size = (block.x+ kernel.size -1) *  (block.x+ kernel.size -1) *sizeof(int);

    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    GPU2DConvolutionTiling_kernel<<<grid, block, shared_memory_size>>>(dev_input.pixels,dev_output.pixels, dev_kernel.pixels,
                                                                       dev_input.width, dev_input.height,kernel.size );
    cudaDeviceSynchronize();

    auto t2 = Clock::now();
    std::chrono::duration<double, std::milli> time = t2 - t1;

    freeImageDev(dev_input);
    copyFromDeviceToHost(dev_output, output);
    freeImageDev(dev_output);

    freeKernelDev(dev_kernel);

    return time.count();
}

__host__ void testConvolution(int img_size, int kernel_size, int block_width, const int& iters,
                              const std::string& input_dir,const std::string& output_dir){
    // outputfile to store the results
    std::ofstream outfile;
    outfile.open("results/Convolution_results.txt", std::ios_base::app);
    if (!outfile.is_open()) {
        printf("Error: Unable to write to Convolution_results.txt \n");
        exit(1);
    }

    std::cout << "Blurring " << "rand" + std::to_string(img_size) + ".png" << " " << "with a " << kernel_size << "x"
              << kernel_size <<" kernel" << std::endl;
    outfile << "Blurring " << "rand" + std::to_string(img_size) + ".png" << " " << "with a " << kernel_size << "x"
            << kernel_size <<" kernel" << std::endl;

    dim3 grid((img_size + block_width - 1) / block_width, (img_size + block_width - 1) / block_width, 1);
    dim3 block(block_width, block_width);


    std::cout << "The size of the square block is: " << block_width << " and the grid contains " << grid.x << "x" <<
              grid.y << "x" << grid.z << " blocks" << std::endl;
    outfile << "The size of the square block is: " << block_width << " and the grid contains " << grid.x << "x" <<
            grid.y << "x" << grid.z << " blocks" << std::endl;
    double mean_time_seq = 0, mean_time_GPU = 0, mean_speedup = 0, mean_time_GPU_tiling = 0, mean_speedup_tiling =0;
    for(int iter=0; iter< iters; iter++) {
        std::cout << "Total iteration: "<< iter+1 << "/" << iters << std::endl;
        Image image = generateImage(img_size, img_size, 11+iter);
        saveImageToFile(image, "rand" + std::to_string(img_size) + ".png", input_dir);
        Image output_image_seq = copyImage(image);
        Image output_image_GPU = copyImage(image);
        Image output_image_GPU_tiling = copyImage(image);
        Kernel kernel = generateBlurKernel(kernel_size);

        double time_seq = timeSequential(image, kernel, output_image_seq);
        mean_time_seq += time_seq;
        saveImageToFile(output_image_seq, "blur_rand" + std::to_string(img_size) + ".png", output_dir);

        double time_GPU = timeGPU(image, kernel, output_image_GPU, grid, block);
        double speedup_GPU = time_seq / time_GPU;
        mean_time_GPU += time_GPU;
        mean_speedup += speedup_GPU;
        saveImageToFile(output_image_GPU, "blur_rand" + std::to_string(img_size) + "_GPU.png", output_dir);

        compareImages(output_image_seq, output_image_GPU, outfile);

        double time_GPU_tiling = timeGPU_tiling(image, kernel, output_image_GPU_tiling, grid, block);
        double speedup_GPU_tiling = time_seq / time_GPU_tiling;
        mean_time_GPU_tiling += time_GPU_tiling;
        mean_speedup_tiling += speedup_GPU_tiling;
        saveImageToFile(output_image_GPU_tiling, "blur_rand" + std::to_string(img_size) + "_GPU_tiling.png",
                        output_dir);

        compareImages(output_image_seq, output_image_GPU_tiling, outfile);

        freeImageHost(image);
        freeImageHost(output_image_seq);
        freeImageHost(output_image_GPU);
    }
    mean_time_seq /= iters, mean_time_GPU /= iters, mean_speedup /= iters,
    mean_time_GPU_tiling /= iters, mean_speedup_tiling /= iters;
    printf("On average, the sequential 2D convolution completed in %fms \n", mean_time_seq);
    outfile << "On average, the sequential 2D convolution completed in " << mean_time_seq << "ms" << std::endl;

    printf("On average, the GPU 2D convolution completed in %fms with a speedup of %f \n", mean_time_GPU, mean_speedup);
    outfile << "On average, the GPU 2D convolution completed in " << mean_time_GPU << "ms with a speedup of " << mean_speedup
            << std::endl;

    printf("On average, the GPU 2D convolution with tiling completed in %fms with a speedup of %f \n", mean_time_GPU_tiling,
           mean_speedup_tiling);
    outfile << "On average, the GPU 2D convolution with tiling completed in " << mean_time_GPU_tiling
            << "ms with a speedup of " << mean_speedup_tiling << std::endl;

    outfile << std::endl;
}

__host__ void testConvolution_kernel(int img_size, int kernel_size, int block_width, const int& iters,
                              const std::string& input_dir,const std::string& output_dir){
    // outputfile to store the results
    std::ofstream outfile;
    outfile.open("results/Convolution_results.txt", std::ios_base::app);
    if (!outfile.is_open()) {
        printf("Error: Unable to write to Convolution_results.txt \n");
        exit(1);
    }

    std::cout << "Blurring " << "rand" + std::to_string(img_size) + ".png" << " " << "with a " << kernel_size << "x"
              << kernel_size <<" non constant kernel" << std::endl;
    outfile << "Blurring " << "rand" + std::to_string(img_size) + ".png" << " " << "with a " << kernel_size << "x"
            << kernel_size <<" non constant kernel" << std::endl;

    dim3 grid((img_size + block_width - 1) / block_width, (img_size + block_width - 1) / block_width, 1);
    dim3 block(block_width, block_width);


    std::cout << "The size of the square block is: " << block_width << " and the grid contains " << grid.x << "x" <<
              grid.y << "x" << grid.z << " blocks" << std::endl;
    outfile << "The size of the square block is: " << block_width << " and the grid contains " << grid.x << "x" <<
            grid.y << "x" << grid.z << " blocks" << std::endl;
    double mean_time_seq = 0, mean_time_GPU = 0, mean_speedup = 0, mean_time_GPU_tiling = 0, mean_speedup_tiling =0;
    for(int iter=0; iter< iters; iter++) {
        std::cout << "Total iteration: "<< iter+1 << "/" << iters << std::endl;
        Image image = generateImage(img_size, img_size, 11+iter);
        saveImageToFile(image, "rand" + std::to_string(img_size) + ".png", input_dir);
        Image output_image_seq = copyImage(image);
        Image output_image_GPU = copyImage(image);
        Image output_image_GPU_tiling = copyImage(image);
        Kernel kernel = generateBlurKernel(kernel_size);

        double time_seq = timeSequential(image, kernel, output_image_seq);
        mean_time_seq += time_seq;
        saveImageToFile(output_image_seq, "blur_rand" + std::to_string(img_size) + ".png", output_dir);

        double time_GPU = timeGPU_kernel(image, kernel, output_image_GPU, grid, block);
        double speedup_GPU = time_seq / time_GPU;
        mean_time_GPU += time_GPU;
        mean_speedup += speedup_GPU;
        saveImageToFile(output_image_GPU, "blur_rand" + std::to_string(img_size) + "_GPU.png", output_dir);

        compareImages(output_image_seq, output_image_GPU, outfile);

        double time_GPU_tiling = timeGPU_tiling_kernel(image, kernel, output_image_GPU_tiling, grid, block);
        double speedup_GPU_tiling = time_seq / time_GPU_tiling;
        mean_time_GPU_tiling += time_GPU_tiling;
        mean_speedup_tiling += speedup_GPU_tiling;
        saveImageToFile(output_image_GPU_tiling, "blur_rand" + std::to_string(img_size) + "_GPU_tiling.png",
                        output_dir);

        compareImages(output_image_seq, output_image_GPU_tiling, outfile);

        freeImageHost(image);
        freeImageHost(output_image_seq);
        freeImageHost(output_image_GPU);
        freeKernelHost(kernel);
    }
    mean_time_seq /= iters, mean_time_GPU /= iters, mean_speedup /= iters,
    mean_time_GPU_tiling /= iters, mean_speedup_tiling /= iters;
    printf("On average, the sequential 2D convolution completed in %fms \n", mean_time_seq);
    outfile << "On average, the sequential 2D convolution completed in " << mean_time_seq << "ms" << std::endl;

    printf("On average, the GPU 2D convolution with non constant kernel completed in %fms with a speedup of %f \n", mean_time_GPU, mean_speedup);
    outfile << "On average, the GPU 2D convolution with non constant kernel  completed in " << mean_time_GPU << "ms with a speedup of " << mean_speedup
            << std::endl;

    printf("On average, the GPU 2D convolution with tiling and non constant kernel completed in %fms with a speedup of %f \n", mean_time_GPU_tiling,
           mean_speedup_tiling);
    outfile << "On average, the GPU 2D convolution with tiling and non constant kernel completed in " << mean_time_GPU_tiling
            << "ms with a speedup of " << mean_speedup_tiling << std::endl;

    outfile << std::endl;
}
int main() {
    // choose cuda device
    cudaSetDevice(1);
    // reminder that 2*block_width*block_width needs to be larger than (block_width+kernel_size-1)*(block_width+kernel_size-1)
    // e.g. if kernel_size = 5 block_width needs to be at least 10
    // max block_width is 32 since they are squares

    // test different image siz
    for(int size=8; size <= 32768; size = size*2){
        testConvolution(size, 5, 16, 5,
                        "Grayscale images", "Output grayscale images");
    }

    // test different block_width
    for(int block_width=10; block_width <= 32; block_width = block_width+2){
        testConvolution(8192, 5, block_width, 5,
                        "Grayscale images", "Output grayscale images");
    }

    // test different kernels
    for(int kernel_size=3; kernel_size <= 9; kernel_size = kernel_size+2){
        testConvolution(8192, kernel_size, 32, 5,
                        "Grayscale images", "Output grayscale images");
    }

    // test different kernels with non-constant kernel
    for(int kernel_size=3; kernel_size <= 9; kernel_size = kernel_size+2){
        testConvolution_kernel(8192, kernel_size, 32, 5,
                        "Grayscale images", "Output grayscale images");
    }
    return 0;
}
