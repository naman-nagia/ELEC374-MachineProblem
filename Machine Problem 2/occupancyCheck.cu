// Written by: Naman Nagia 
// Student #: 20357592 
//Main Source: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Declartion of Kernel
__global__ void matMulTiled(const float* M, const float* N, float* P, int width, int TILE_WIDTH);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA device found.\n");
        return 0;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n\n", prop.maxThreadsPerMultiProcessor);

    // Tile sizes for which the stats will be outputted
    int tileWidths[] = { 2, 4, 8, 16, 32 };
    int numTileWidths = sizeof(tileWidths) / sizeof(int);

    // Loop through tile widths and print stats
    for (int i = 0; i < numTileWidths; i++)
    {
        int tileWidth = tileWidths[i];
        int blockSize = tileWidth * tileWidth;

        size_t sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);

        // Occupancy API
        int maxActiveBlocks = 0;
        cudaError_t occErr = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            matMulTiled,
            blockSize,
            sharedMemSize
        );

        printf("============================================\n");
        printf("TILE_WIDTH = %d\n", tileWidth);
        printf("BlockSize (threads) = %d\n", blockSize);
        printf("Dynamic Shared Memory (bytes) = %zu\n", sharedMemSize);

        if (occErr == cudaSuccess)
        {
            int threadsPerSM = maxActiveBlocks * blockSize;
            int totalThreads = threadsPerSM * prop.multiProcessorCount;

            printf("Occupancy Info:\n");
            printf("  Max active blocks per SM : %d\n", maxActiveBlocks);
            printf("  => Threads per SM        : %d\n", threadsPerSM);
            printf("  => Total Threads on GPU  : %d\n", totalThreads);
        }
        else {
            printf("Error from cudaOccupancyMaxActiveBlocksPerMultiprocessor: %s\n",
                cudaGetErrorString(occErr));
        }

        // Resource usage from cudaFuncGetAttributes
        cudaFuncAttributes funcAttr;
        cudaFuncGetAttributes(&funcAttr, matMulTiled);

        printf("Kernel Resource Usage:\n");
        printf("  Registers per thread : %d\n", funcAttr.numRegs);
        printf("  Static Shared Mem    : %zd bytes\n", funcAttr.sharedSizeBytes); // likely remains 0 since its allocated dynamically
        printf("  Local Mem           : %zd bytes\n", funcAttr.localSizeBytes);
        printf("  Const Mem           : %zd bytes\n", funcAttr.constSizeBytes);
        printf("\n");
    }

    return 0;
}