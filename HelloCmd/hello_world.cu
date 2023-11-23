#include<cuda.h>
#include<cuda_runtime.h>

#include<iostream>


__global__ void cuda_hello(){
    printf("Hello World from GPU and Im pahuldeep!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
