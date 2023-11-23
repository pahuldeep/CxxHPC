
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
using namespace std;

int main() {
	int device_count = 0;

	if (device_count == cudaGetDevice(&device_count)) printf("No device");
	else printf("device avail");

	printf("%d", cudaGetDevice(&device_count));

	cudaDeviceSynchronize();

}
