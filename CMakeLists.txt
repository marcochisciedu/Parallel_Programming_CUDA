cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)

project(Parallel_Programming_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Parallel_Programming_CUDA main.cu
        Image.cu
        Image.cuh
        SequentialConvolution.cu
        SequentialConvolution.cuh
        GPUConvolution.cu
        GPUConvolution.cuh
)


set_target_properties(Parallel_Programming_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
