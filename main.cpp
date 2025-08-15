
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void kernel(int* device_a, const int* device_b, const int* device_c, const int size)
{
	int index = threadIdx.x;
	if (index >= size)
		return;
	device_a[index] = device_b[index] + device_c[index];
}

void runTestKernel()
{
	int size = 15;

	int* host_a = new int[size];
	int* host_b = new int[size];
	int* host_c = new int[size];

	int* device_a = nullptr;
	int* device_b = nullptr;
	int* device_c = nullptr;

	cudaMalloc((void**)&device_a, sizeof(int) * size);
	cudaMalloc((void**)&device_b, sizeof(int) * size);
	cudaMalloc((void**)&device_c, sizeof(int) * size);

	for (int i = 0; i < size; i++)
	{
		host_a[i] = 0;
		host_b[i] = i;
		host_c[i] = i * 10;
	}

	cudaMemcpy(device_a, host_a, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, sizeof(int) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_c, host_c, sizeof(int) * size, cudaMemcpyHostToDevice);

	kernel <<<1, size>>> (device_a, device_b, device_c, size);
	cudaDeviceSynchronize();

	cudaMemcpy(host_a, device_a, sizeof(int) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_b, device_b, sizeof(int) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, device_c, sizeof(int) * size, cudaMemcpyDeviceToHost);

	std::cout << "A = B + C: "; for (int i = 0; i < size; i++) { std::cout << host_a[i] << " "; }
	std::cout <<       "\nB: "; for (int i = 0; i < size; i++) { std::cout << host_b[i] << " "; }
	std::cout <<       "\nC: "; for (int i = 0; i < size; i++) { std::cout << host_c[i] << " "; }

	delete[] host_a;
	delete[] host_b;
	delete[] host_c;

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
}

int main()
{
	runTestKernel();
	return 0;
}

