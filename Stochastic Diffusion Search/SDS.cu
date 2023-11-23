#include <iostream>
#include <string>
#include <curand_kernel.h>

using namespace std;

const int N = 10;
const int maxIter = 30;

__device__ float r(curandState* state) {
    // Generate a random number in the range [0, 1] using CURAND
    return curand_uniform(state);
}

__global__ void sdsKernel(char* ss, char* model, int ssLength, int modelLength, int* hypo, bool* status) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(clock64(), tid, 0, &state); // Initialize the random state for this thread

    if (tid < N) {
        int hypoValue = hypo[tid];
        bool agentStatus = false;

        for (int itr = 0; itr < maxIter; itr++) {
            int microFeature = r(&state) * modelLength;

            if (ss[hypoValue + microFeature] == model[microFeature]) {
                agentStatus = true;
            } else {
                agentStatus = false;
            }

            __syncthreads();

            if (!agentStatus) {
                int randAgent = r(&state) * N;
                if (status[randAgent]) {
                    hypoValue = hypo[randAgent];
                } else {
                    hypoValue = r(&state) * (ssLength - modelLength);
                }
            } else {
                microFeature = r(&state) * modelLength;
                if (ss[hypoValue + microFeature] == model[microFeature]) {
                    agentStatus = true;
                } else {
                    agentStatus = false;
                }
            }

            __syncthreads();
        }
        hypo[tid] = hypoValue;
        status[tid] = agentStatus;
    }
}

int main() {
    string ss = "try to find sds in this sentence";
    string model = " sds ";

    int ssLength = ss.length();
    int modelLength = model.length();

    char* d_ss;
    char* d_model;
    int* d_hypo;
    bool* d_status;

    // Allocate device memory
    cudaMalloc((void**)&d_ss, ssLength);
    cudaMalloc((void**)&d_model, modelLength);
    cudaMalloc((void**)&d_hypo, N * sizeof(int));
    cudaMalloc((void**)&d_status, N * sizeof(bool));

    // Copy data from host to device
    cudaMemcpy(d_ss, ss.c_str(), ssLength, cudaMemcpyHostToDevice);
    cudaMemcpy(d_model, model.c_str(), modelLength, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sdsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ss, d_model, ssLength, modelLength, d_hypo, d_status);

    // Copy results from device to host
    int* hypo = new int[N];
    bool* status = new bool[N];
    cudaMemcpy(hypo, d_hypo, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(status, d_status, N * sizeof(bool), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_ss);
    cudaFree(d_model);
    cudaFree(d_hypo);
    cudaFree(d_status);

    for (int i = 0; i < N; i++) {
        cout << "Agent " << i << " - Found: " << ss.substr(hypo[i], modelLength) << " Status: " << status[i] << endl;
    }

    return 0;
}
