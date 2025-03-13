#include <stdio.h>          
#include <stdlib.h>         
#include <cuda_runtime.h>   
#include <math.h>

// Kernel for matrix multiplication on the GPU
__global__ void matMulKernel(const float* M, const float* N, float* P, int width) {
    // Compute row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Out of Bounds check
    if (row < width && col < width) {
        float sum = 0.0f;
        // Perform multiplication
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        // Store the result
        P[row * width + col] = sum;
    }
}

// Function for matrix multiplication on the CPU
void matMulCPU(const float* M, const float* N, float* P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            // Multiply row i of M by column j of N
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            // Store the result in P
            P[i * width + j] = sum;
        }
    }
}

// Comparison function to check correct answer
bool compareArrays(const float* A, const float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        // If difference > tolerance, return false
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
// Kernel execution time (data transfer not included)
void measureKernelTime(const float* dM, const float* dN, float* dP, 
    int width, int blockWidth, float &kernelTimeMs) {
    // Create the dim3 block and grid
    dim3 block(blockWidth, blockWidth);
    dim3 grid((width + blockWidth - 1) / blockWidth, (width + blockWidth - 1) / blockWidth);

    // Create CUDA events to record time taken
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event, launch kernel
    cudaEventRecord(start);
    matMulKernel<<<grid, block>>>(dM, dN, dP, width);
    cudaEventRecord(stop);

    // Synchronize to make sure kernel finished, measure time
    // Was causing error (may not be needed)
    cudaEventSynchronize(stop);

    // Calculate final value
    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    // Free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main()
{
    // Part 2.1: Transfer Times
    const int sizes1[] = {256, 512, 1024, 2048, 4096};
    const int numSizes1 = sizeof(sizes1) / sizeof(int);

    printf("--- Part 2.1: H to D and D to H Transfer Times ---\n");
    printf("MatrixSizes: 256, 512, 1024, 2048, 4096\n\n");

    // Arrays to store transfer times
    float hToDTimes[numSizes1];
    float dToHTimes[numSizes1];

    // Iterate through varying sizes
    for (int idx = 0; idx < numSizes1; idx++) {
        int width = sizes1[idx];
        size_t bytes = width * (size_t)width * sizeof(float);

        // Allocate host memory
        float* hM = (float*)malloc(bytes);
        float* hN = (float*)malloc(bytes);
        float* hP = (float*)malloc(bytes);

        // Initialize data
        srand(0);
        for (int i = 0; i < width * width; i++) {
            hM[i] = (float)(rand() % 10);
            hN[i] = (float)(rand() % 10);
        }

        // Allocate device memory
        float* dM; float* dN; float* dP;
        cudaMalloc((void**)&dM, bytes);
        cudaMalloc((void**)&dN, bytes);
        cudaMalloc((void**)&dP, bytes);

        // Measure Host to Device transfer time
        cudaEvent_t startHtoD, stopHtoD;
        cudaEventCreate(&startHtoD);
        cudaEventCreate(&stopHtoD);
        cudaEventRecord(startHtoD);
        cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopHtoD);
        cudaEventSynchronize(stopHtoD); // for error

        float timeHtoD = 0.0f;
        cudaEventElapsedTime(&timeHtoD, startHtoD, stopHtoD);
        hToDTimes[idx] = timeHtoD;

        // Measure Device to Host transfer time
        // Typically you'd measure for P but using N and M for simplicity
        cudaEvent_t startDtoH, stopDtoH;
        cudaEventCreate(&startDtoH);
        cudaEventCreate(&stopDtoH);
        cudaEventRecord(startDtoH);
        cudaMemcpy(hP, dM, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hP, dN, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopDtoH);
        cudaEventSynchronize(stopDtoH);

        float timeDtoH = 0.0f;
        cudaEventElapsedTime(&timeDtoH, startDtoH, stopDtoH);
        dToHTimes[idx] = timeDtoH;

        // Free memory 
        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        free(hM);
        free(hN);
        free(hP);
        cudaEventDestroy(startHtoD);
        cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startDtoH);
        cudaEventDestroy(stopDtoH);
    }

    // Print results for experiment (1)
    printf("Host to Device Transfer Times (ms) by Matrix Size:\n");
    for (int i = 0; i < numSizes1; i++) {
        printf("  Size %d x %d : %f ms\n", sizes1[i], sizes1[i], hToDTimes[i]);
    }
    printf("\nDevice to Host Transfer Times (ms) by Matrix Size:\n");
    for (int i = 0; i < numSizes1; i++) {
        printf("  Size %d x %d : %f ms\n", sizes1[i], sizes1[i], dToHTimes[i]);
    }
    printf("\n");

    // Part 2.2
    const int sizes2[] = {256, 512, 1024};
    const int numSizes2 = sizeof(sizes2) / sizeof(int);

    // Code below was copied from the code snippet above:
    // Thus refer to comments above for code below
    printf("--- Part 2.2 : CPU vs. GPU (Single Thread) ---\n");
    printf("MatrixSizes: 256, 512, 1024\n\n");

    for (int idx = 0; idx < numSizes2; idx++) {
        int width = sizes2[idx];
        size_t bytes = width * (size_t)width * sizeof(float);

        float* hM = (float*)malloc(bytes);
        float* hN = (float*)malloc(bytes);
        float* hP = (float*)malloc(bytes);
        float* hRef = (float*)malloc(bytes);

        srand(0);
        for (int i = 0; i < width * width; i++) {
            hM[i] = (float)(rand() % 10);
            hN[i] = (float)(rand() % 10);
        }

        float *dM, *dN, *dP;
        cudaMalloc((void**)&dM, bytes);
        cudaMalloc((void**)&dN, bytes);
        cudaMalloc((void**)&dP, bytes);

        cudaEvent_t startHtoD, stopHtoD;
        cudaEventCreate(&startHtoD);
        cudaEventCreate(&stopHtoD);
        cudaEventRecord(startHtoD);
        cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopHtoD);
        cudaEventSynchronize(stopHtoD);

        float hToD_ms = 0.0f;
        cudaEventElapsedTime(&hToD_ms, startHtoD, stopHtoD);

        // Assign only 1 block with 1 thread
        dim3 block(1, 1);
        dim3 grid(1, 1);

        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);

        // GPU
        cudaEventRecord(startKernel);
        matMulKernel<<<grid, block>>>(dM, dN, dP, width);
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);

        float gpuKernel_ms = 0.0f;
        cudaEventElapsedTime(&gpuKernel_ms, startKernel, stopKernel);

        cudaEvent_t startDtoH, stopDtoH;
        cudaEventCreate(&startDtoH);
        cudaEventCreate(&stopDtoH);

        cudaEventRecord(startDtoH);
        cudaMemcpy(hP, dP, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopDtoH);
        cudaEventSynchronize(stopDtoH);

        float dToH_ms = 0.0f;
        cudaEventElapsedTime(&dToH_ms, startDtoH, stopDtoH);

        // CPU time
        clock_t cpuStart = clock();
        matMulCPU(hM, hN, hRef, width);
        clock_t cpuEnd = clock();
        float cpu_ms = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

        // Getting values for Ignoring transfer vs. including transfer
        float gpuTotalNoTransfer = gpuKernel_ms;               // GPU only kernel
        float gpuTotalWithTransfer = gpuKernel_ms + hToD_ms + dToH_ms;

        // Compare correctness tolerance chosen manually
        bool pass = compareArrays(hRef, hP, width * width, 1e-3f);

        printf("Matrix Size %d x %d\n", width, width);
        printf("  CPU Time (ms)                    : %f\n", cpu_ms);
        printf("  GPU Time (1 block,1 thread) (ms) : %f (NO Transfer), %f (WITH Transfer)\n",
                gpuTotalNoTransfer, gpuTotalWithTransfer);
        printf("  => Transfer Times: H->D = %f ms, D->H = %f ms\n", hToD_ms, dToH_ms);
        printf("  => %s\n\n", pass ? "Test PASSED" : "Test FAILED");

        // Clean up
        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        free(hM);
        free(hN);
        free(hP);
        free(hRef);
        cudaEventDestroy(startHtoD);
        cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);
        cudaEventDestroy(stopDtoH);
    }

    // Part 3.3: Vary block width and only measure kernel times
    printf("--- Part 3.3: Kernel Times with varying Matrix Size & Block Width---\n");
    printf("MatrixSizes: 256, 512, 1024, 2048, 4096  |  BlockWidth: 2,4,8,16,32\n\n");

    int blockWidths[5] = {2, 4, 8, 16, 32};
    int sizes3[] = {256, 512, 1024, 2048, 4096};
    int numSizes3 = 5;

    // Print a table of times for each (matrixSize, blockWidth) using for loop for ease
    for (int bwIdx = 0; bwIdx < 5; bwIdx++) {
        int bWidth = blockWidths[bwIdx];
        printf("BlockWidth = %d\n", bWidth);
        for (int sIdx = 0; sIdx < numSizes3; sIdx++) {
            int width = sizes3[sIdx];
            size_t bytes = width * (size_t)width * sizeof(float);

            float* hM = (float*)malloc(bytes);
            float* hN = (float*)malloc(bytes);
            float* hP = (float*)malloc(bytes);

            srand(0);
            for (int i = 0; i < width * width; i++) {
                hM[i] = (float)(rand() % 10);
                hN[i] = (float)(rand() % 10);
            }

            float *dM, *dN, *dP;
            cudaMalloc((void**)&dM, bytes);
            cudaMalloc((void**)&dN, bytes);
            cudaMalloc((void**)&dP, bytes);

            // Transfer time is ignored
            cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);

            // Measure kernel time
            float kernelMs = 0.0f;
            measureKernelTime(dM, dN, dP, width, bWidth, kernelMs);

            printf("  Size %d x %d -> Kernel Time = %f ms\n", width, width, kernelMs);

            // Free memory
            cudaFree(dM);
            cudaFree(dN);
            cudaFree(dP);
            free(hM);
            free(hN);
            free(hP);
        }
        printf("\n");
    }

    return 0;
}
