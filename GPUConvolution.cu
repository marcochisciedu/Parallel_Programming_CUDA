
#include "GPUConvolution.cuh"
// constant memory kernel
#define MAX_KERNEL_DIM (10*10)
extern __constant__ float  kernel_pixels[MAX_KERNEL_DIM];

//convolution with GPU
__global__ void GPU2DConvolution(const int* input_img,  int* output_img,
                                 int width, int height, int kernel_size){
    // get current thread position on the whole image
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check if thread is within image's borders
    if(col < width && row < height){
        float pixel_value = 0;

        // where the upper left value of the kernel is located
        int start_row = row - (kernel_size/2);
        int start_col = col - (kernel_size/2);

        // iterate through the kernel
        for(int i=0; i < kernel_size; i++){
            for(int j=0; j <kernel_size;j++){
                int cur_row = start_row +i;
                int cur_col = start_col +j;
                // check if the kernel is within image's borders
                if(cur_row >=0 && cur_col >=0 && cur_row < height && cur_col < width){
                    // convolution
                    pixel_value += static_cast<float>(input_img[cur_row* width + cur_col]) * kernel_pixels[i*kernel_size+j];
                }
            }
        }
        output_img[row*width+col]= static_cast<int>(lround(pixel_value) );
    }
}

// convolution with tiling
__global__ void GPU2DConvolutionTiling(const int* input_img,  int* output_img,
                                 int width, int height, int kernel_size){
    // blocks and image are square
    int tiling_width = blockDim.x + kernel_size -1;

    extern __shared__ int shared_input_image[];

    // first batch loading, get the coordinates of all the needed pixels starting with the upper left corner
    int dest = threadIdx.y * blockDim.x + threadIdx.x;
    int destY = dest / tiling_width, destX = dest % tiling_width;
    int srcY = blockIdx.y * blockDim.x + destY - (kernel_size/2);
    int srcX = blockIdx.x * blockDim.x + destX - (kernel_size/2);
    int src = srcY * width + srcX;
    // check image's border
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        shared_input_image[destY * tiling_width+destX] = input_img[src];
    else
        shared_input_image[destY * tiling_width+destX] = 0;

    // second batch loading, start with a blockDim*blockDim offset
    dest = threadIdx.y *  blockDim.x + threadIdx.x +  blockDim.x *  blockDim.x;
    destY = dest / tiling_width, destX = dest % tiling_width;
    srcY = blockIdx.y *  blockDim.x + destY - (kernel_size/2);
    srcX = blockIdx.x *  blockDim.x + destX - (kernel_size/2);
    src = srcY * width + srcX;
    // check if we're still within the tiling area
    if (destY < tiling_width) {
        // check image's border
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            shared_input_image[destY * tiling_width + destX] = input_img[src];
        else
            shared_input_image[destY * tiling_width + destX] = 0;
    }

    // wait until all the threads loaded their pixels into the shared memory
    __syncthreads();

    // get current position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check image borders
    if(col < width && row < height){
        float pixel_value = 0;
        // iterate through the kernel, no need to check if valid (loaded 0 as value)
        for(int i=0; i < kernel_size; i++){
            for(int j=0; j <kernel_size;j++){
                //convolution
                pixel_value += static_cast<float>(shared_input_image[(threadIdx.y + i)*tiling_width+(threadIdx.x +j)])
                        * kernel_pixels[i*kernel_size+j];
            }
        }
        output_img[row*width+col]= static_cast<int>(lround(pixel_value) );
    }
}