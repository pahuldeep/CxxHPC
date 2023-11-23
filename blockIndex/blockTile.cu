#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 255

__global__ void print(void) {
    int i = blockIdx.x;
    printf("%d", &i);
}

int main()
{
    print << <1, 1 >> > ();

}
