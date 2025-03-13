#include <cuda_runtime.h>
#include <stdio.h>

int getCUDACores(cudaDeviceProp p) {
    int coresPerSM = 0;

    // Determine the number of cores per SM based on compute capability information was found at the link below
    // https://developer.nvidia.com/cuda-gpus
    if (p.major == 2) {
        if (p.minor == 1) coresPerSM = 48;
        else coresPerSM = 32;
    }
    else if (p.major == 3) {
        coresPerSM = 192;
    }
    else if (p.major == 5) {
        coresPerSM = 128;
    }
    else if (p.major == 6) {
        if (p.minor == 1 || p.minor == 2) coresPerSM = 128;
        else coresPerSM = 64;
    }
    else if (p.major == 7) {
        coresPerSM = 64;
    }
    else if (p.major == 8) {
        if (p.minor == 0) coresPerSM = 64;
        else coresPerSM = 128;
    }
    else if (p.major == 9) {
        coresPerSM = 128;
    }
    // not including blackwell GPU cause they are too
    else {
        printf("Unknown CUDA architecture: Compute Capability %d.%d\n", p.major, p.minor);
        return 0;
    }

    return p.multiProcessorCount * coresPerSM;
}

int main() {
    int nd;
    cudaGetDeviceCount(&nd);
    printf("Number of CUDA Devices: %d\n", nd);

    if (nd == 0) {
        printf("No CUDA devices found. Exiting...\n");
        return 1;
    }

    for (int i = 0; i < nd; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);

        printf("\nDevice %d: %s\n", i, p.name);
        printf("Compute Capability: %d.%d\n", p.major, p.minor);
        printf("Clock Rate: %.2f MHz\n", p.clockRate / 1000.0f);
        printf("Number of Streaming Multiprocessors (SMs): %d\n", p.multiProcessorCount);

        int cudaCores = getCUDACores(p);
        if (cudaCores > 0)
            printf("Total CUDA Cores (approx.): %d\n", cudaCores);
        else
            printf("Could not determine CUDA cores for this device.\n");

        printf("Warp Size: %d\n", p.warpSize);
        printf("Global Memory: %.2f GB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Constant Memory: %.2f KB\n", p.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", p.sharedMemPerBlock / 1024.0);
        printf("Registers per Block: %d\n", p.regsPerBlock);
        printf("Max Threads per Block: %d\n", p.maxThreadsPerBlock);
        printf("Max Block Dimensions: (%d, %d, %d)\n", 
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("Max Grid Dimensions: (%d, %d, %d)\n", 
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    }

    return 0;
}


/*
Output for Part 2:
=== Experiment (1): H->D and D->H Transfer Times ===
MatrixSizes: 256, 512, 1024, 2048, 4096

Host->Device Transfer Times (ms) by Matrix Size:
  Size 256 x 256 : 0.255392 ms
  Size 512 x 512 : 0.594688 ms
  Size 1024 x 1024 : 1.745568 ms
  Size 2048 x 2048 : 5.719136 ms
  Size 4096 x 4096 : 23.076544 ms

Device->Host Transfer Times (ms) by Matrix Size:
  Size 256 x 256 : 0.279648 ms
  Size 512 x 512 : 0.580544 ms
  Size 1024 x 1024 : 1.581696 ms
  Size 2048 x 2048 : 5.379104 ms
  Size 4096 x 4096 : 22.408545 ms

=== Experiment (2): CPU vs. GPU (Single Block/Thread) ===
MatrixSizes: 256, 512, 1024

Matrix Size 256 x 256
  CPU Time (ms)                    : 41.000000
  GPU Time (1 block,1 thread) (ms) : 0.561184 (NO Transfer), 0.859296 (WITH Transfer)
  => Transfer Times: H->D = 0.152736 ms, D->H = 0.145376 ms
  => Test FAILED

Matrix Size 512 x 512
  CPU Time (ms)                    : 469.000000
  GPU Time (1 block,1 thread) (ms) : 0.972160 (NO Transfer), 1.872288 (WITH Transfer)
  => Transfer Times: H->D = 0.566752 ms, D->H = 0.333376 ms
  => Test FAILED

Matrix Size 1024 x 1024
  CPU Time (ms)                    : 3829.000000
  GPU Time (1 block,1 thread) (ms) : 1.804672 (NO Transfer), 4.150592 (WITH Transfer)
  => Transfer Times: H->D = 1.553504 ms, D->H = 0.792416 ms
  => Test FAILED

=== Experiment (3): Kernel Times vs. Block Width & Matrix Size ===
MatrixSizes: 256, 512, 1024, 2048, 4096  |  BlockWidth: 2,4,8,16,32

BlockWidth = 2
  Size 256 x 256 -> Kernel Time = 39.539745 ms
  Size 512 x 512 -> Kernel Time = 244.076828 ms
  Size 1024 x 1024 -> Kernel Time = 1260.679688 ms
  Size 2048 x 2048 -> Kernel Time = 10064.927734 ms
  Size 4096 x 4096 -> Kernel Time = 81669.343750 ms

BlockWidth = 4
  Size 256 x 256 -> Kernel Time = 4.161600 ms
  Size 512 x 512 -> Kernel Time = 37.470753 ms
  Size 1024 x 1024 -> Kernel Time = 337.502045 ms
  Size 2048 x 2048 -> Kernel Time = 2664.062744 ms
  Size 4096 x 4096 -> Kernel Time = 21924.910156 ms

BlockWidth = 8
  Size 256 x 256 -> Kernel Time = 1.681600 ms
  Size 512 x 512 -> Kernel Time = 11.526624 ms
  Size 1024 x 1024 -> Kernel Time = 88.355331 ms
  Size 2048 x 2048 -> Kernel Time = 733.556824 ms
  Size 4096 x 4096 -> Kernel Time = 5946.437012 ms

BlockWidth = 16
  Size 256 x 256 -> Kernel Time = 1.331168 ms
  Size 512 x 512 -> Kernel Time = 11.759936 ms
  Size 1024 x 1024 -> Kernel Time = 86.569923 ms
  Size 2048 x 2048 -> Kernel Time = 738.487976 ms
  Size 4096 x 4096 -> Kernel Time = 5909.993652 ms

BlockWidth = 32
  Size 256 x 256 -> Kernel Time = 1.435968 ms
  Size 512 x 512 -> Kernel Time = 11.847040 ms
  Size 1024 x 1024 -> Kernel Time = 95.520798 ms
  Size 2048 x 2048 -> Kernel Time = 742.589905 ms
  Size 4096 x 4096 -> Kernel Time = 5945.530762 ms
*/