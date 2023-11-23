
#include "cuda_runtime.h"
#include "cuda.h"

#include "iostream"
using namespace std;

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "Device name: " << prop.name << "\n";
        cout << "Memory Clock Rate (MHz): " << prop.memoryClockRate / 1024 << "\n";
        cout << "Memory Bus Width (bits): " << prop.memoryBusWidth;
        cout << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        cout << "Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0;
 
        cout << "Shared memory per block(Kbytes) % .1f\n", (float)(prop.sharedMemPerBlock) / 1024.0;
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");

    }
}