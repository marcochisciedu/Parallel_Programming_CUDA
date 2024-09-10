
#include <cmath>

#include "SequentialConvolution.cuh"

// sequential implementation of the 2D convolution
__host__ void sequential2DConvolution(const int* input_img, const float* kernel, int* output_img,
                                      int width, int height, int kernel_size){
    // for each pixel calculate new pixel value
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
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
                        pixel_value += static_cast<float>(input_img[cur_row* width + cur_col] )* kernel[i*kernel_size+j];
                    }
                }
            }
            output_img[row*width+col]= static_cast<int>(std::lround(pixel_value) );
        }
    }
}