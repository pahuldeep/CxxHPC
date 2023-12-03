#include "hello.h"

__global__ void hello(){

    printf("GPU:: Hello World\n");
}

void show_hello(void){

    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    printf("DONE\n");
}
