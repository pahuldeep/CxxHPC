#include <stdio.h>
#include <stdlib.h>

#define N 512

__global__ void device_add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void host_add(int *a, int *b, int *c){
    for(int i = 0; i < N; i++){
        c[i] = a[i] + a[i];
    }
}

void print(int *a, int *b, int *c){
    for(int i = 0; i<N; i++){
        printf("\n %d + %d = %d", a[i], b[i], c[i]);
    }
}

void fill(int *data){
    for(int i=0; i<N; i++){
        data[i] = i;
    }
}

int main(){
    int *a, *b, *c;
    int *device_a, *device_b, *device_c;

    int size = N*sizeof(int);

    a = (int *)malloc(size); fill(a);
    b = (int *)malloc(size); fill(b);
    c = (int *)malloc(size),
    // host_add(a, b, c);

    cudaMalloc((void *)&device_a, N * sizeof(int));
    cudaMalloc((void *)&device_b, N * sizeof(int));
    cudaMalloc((void *)&device_c, N * sizeof(int));

    cudaMemcpy(device_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    device_add<<<N, 1>>>(device_a, device_b, device_c);

    cudaMemcpy(c, device_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print(a, b, c);

    free(a); free(b); free(c);
    cudaFree(device_a); cudaFree(device_b); cudaFree(device_c);
}