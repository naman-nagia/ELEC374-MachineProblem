// Written by: Naman Nagia 
// Student #: 20357592 

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

__global__ void matMulTiled(const float* M, const float* N, float* P, int width, int TILE_WIDTH)
{
    // Shared memory for one tile of M and one tile of N
    extern __shared__ float shared[]; 
    float* tileM = &shared[0];
    float* tileN = &shared[TILE_WIDTH * TILE_WIDTH];

    // Final output matrix row and col
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Partial sum accumulator
    float val = 0.0f;

    // Total tiles loaded
    int numPhases = (width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < numPhases; phase++){
        int tiledCol = phase * TILE_WIDTH + threadIdx.x;
        int tiledRow = phase * TILE_WIDTH + threadIdx.y;

        if (row < width && tiledCol < width)
            tileM[threadIdx.y * TILE_WIDTH + threadIdx.x] = M[row * width + tiledCol];
        else
            tileM[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        if (col < width && tiledRow < width)
            tileN[threadIdx.y * TILE_WIDTH + threadIdx.x] = N[tiledRow * width + col];
        else
            tileN[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0.0f;

        // Synchronize to avoid race conditions
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++){
            val += tileM[threadIdx.y * TILE_WIDTH + k] * tileN[k * TILE_WIDTH + threadIdx.x];
        }

        // Synchronize again to ensure all threads are done writing
        __syncthreads();
    }

    if (row < width && col < width){
        P[row * width + col] = val;
    }
}

// CPU Matrix Multiplicaiton
void matMulCPU(const float* M, const float* N, float* P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

bool compareArrays(const float* A, const float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void measureTiledKernelTime(const float* dM, const float* dN, float* dP, int width, int TILE_WIDTH, float &kernelTimeMs) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Dynamic Shared memory, need 2 tiles: M tile + N tile, each is TILE_WIDTH * TILE_WIDTH
    size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    cudaEventRecord(start);

    matMulTiled<<<grid, block, sharedMemSize>>>(dM, dN, dP, width, TILE_WIDTH);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{   
    // Setting constants
    int matrixSizes[] = {256, 512, 1024, 2048, 4096};
    int numSizes = sizeof(matrixSizes) / sizeof(int);

    int tileWidths[] = {2, 4, 8, 16, 32};
    int numTileWidths = sizeof(tileWidths) / sizeof(int);

    printf("GPU Shared Memory (Tiled) Matrix Multiplication\n");
    printf("Measuring Kernel Time Only, Excluding Data Transfer.\n\n");

    // Runs each tile width
    for (int tIdx = 0; tIdx < numTileWidths; tIdx++)
    {
        int TILE_WIDTH = tileWidths[tIdx];
        printf("=============================================\n");
        printf("TILE_WIDTH = %d\n\n", TILE_WIDTH);

        // Runs each matrix size
        for (int sIdx = 0; sIdx < numSizes; sIdx++)
        {
            int width = matrixSizes[sIdx];
            size_t bytes = (size_t)width * width * sizeof(float);

            float* hM   = (float*) malloc(bytes);
            float* hN   = (float*) malloc(bytes);
            float* hP   = (float*) malloc(bytes);  // GPU result
            float* hRef = (float*) malloc(bytes);  // CPU result (reference)

            // Randomly generate values to initlize matricies
            srand(0);
            for (int i = 0; i < width * width; i++) {
                hM[i] = (float)(rand() % 10);
                hN[i] = (float)(rand() % 10);
            }

            float *dM, *dN, *dP;
            cudaMalloc((void**)&dM, bytes);
            cudaMalloc((void**)&dN, bytes);
            cudaMalloc((void**)&dP, bytes);

            // Transfer data from Host to Device
            cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);

            // Measure tiled kernel time
            float kernelMs = 0.0f;
            measureTiledKernelTime(dM, dN, dP, width, TILE_WIDTH, kernelMs);

            // Transfer result back from Device to Host
            cudaMemcpy(hP, dP, bytes, cudaMemcpyDeviceToHost);

            // Compute CPU refernce
            matMulCPU(hM, hN, hRef, width);

            bool pass = compareArrays(hRef, hP, width * width, 1e-3f);

            printf("Matrix Size %d x %d\n", width, width);
            printf("  Kernel Time  : %f ms\n", kernelMs);
            if (pass){
                printf("  Result Check : Test PASSED\n\n");
            } else{
                printf("  Result Check : Test FAILED\n\n");
            }

            cudaFree(dM);
            cudaFree(dN);
            cudaFree(dP);

            free(hM);
            free(hN);
            free(hP);
            free(hRef);
        }
    }
    return 0;
}