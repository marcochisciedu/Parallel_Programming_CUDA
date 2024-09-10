
#include "GPUConvolution.cuh"
#define MAX_KERNEL_DIM (10*10)
extern __constant__ float  kernel_pixels[MAX_KERNEL_DIM];

__global__ void GPU2DConvolution(const int* input_img,  int* output_img,
                                 int width, int height, int kernel_size){
    // get current position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // check image borders
    if(col < width && row < height){
        float pixel_value = 0;

        // where the upper left value of the kernel is located
        int start_row = row - (kernel_size/2);
        int start_col = col - (kernel_size/2);

        for(int i=0; i < kernel_size; i++){
            for(int j=0; j <kernel_size;j++){
                int cur_row = start_row +i;
                int cur_col = start_col +j;
                // check borders
                if(cur_row >=0 && cur_col >=0 && cur_row < height && cur_col < width){
                    // convolution
                    pixel_value += static_cast<float>(input_img[cur_row* width + cur_col]) * kernel_pixels[i*kernel_size+j];
                }
            }
        }
        output_img[row*width+col]= static_cast<int>(lround(pixel_value) );
    }
}

__global__ void GPU2DConvolutionTiling(const int* input_img,  int* output_img,
                                 int width, int height, int kernel_size){
    // output image tile width for a square block
    int out_tile_width = blockDim.x - kernel_size +1;

    //modificare da main come codice?
    __shared__ int shared_input_image[16][ 16];
}