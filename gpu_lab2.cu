#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// zad 1
__global__ void kernel() // kernel, to be executed on GPU
{
	// print the blocks and threads IDs
	// warp = 32 threads. (128/32 = threads per block)
	int warpIdValue = 0;
	warpIdValue = threadIdx.x / 32;

	printf(
		"\nThe block ID is %d --- The thread ID is %d --- The Wardp ID %d",
		blockIdx.x,
		threadIdx.x,
		warpIdValue
	);
}
// zad2
__global__ void kernel_dwa(int a)
{
	int warpIdValue = 0;
	warpIdValue = threadIdx.x / 32;
	int wynik = warpIdValue * a;
	printf("\n%d", wynik);
}
// zad3
__global__ void kernel_czy_podzielny_przez_trzy()
{
	if (threadIdx.x % 3 == 0)
	{
		printf("\nId watku %d jest podzielne przez 3", threadIdx.x);
	}
	else
	{
		printf("\nId watku %d nie jest podzielne przez 3", threadIdx.x);
	}
}
// zad4
__global__ void kernel100_index_blok()
{
	int a = 100;
	int wynik = a + threadIdx.x + blockIdx.x;
	printf("\n%d", wynik);
}
// zad5
__global__ void kernel_index_warp_parzyste()
{
	int warpIdValue = threadIdx.x / 32;
	int a = threadIdx.x + blockIdx.x + warpIdValue;
	if (a % 2 == 0)
	{
		printf("\n Suma identyfikatorow %d jest parzysta", a);
	}
	else
	{
		printf("\n Suma identyfikatorow %d jest nieparzysta", a);
	}
}
// zad6
__global__ void kernel_roznica_tab(int n, float* a, float* b, float* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		c[i] = a[i] - b[i];
	}
}
// zad7
/*__global__ void parzyste_nieparzyste_tab(int n, float* a, float* b, float* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i % 2 = 0)
	{
		c[i] = a[i] + b[i];
	}
	else
	{
		c[i] = a[i] - b[i];
	}
}*/

int main() // function, execute on CPU
{
	// format: <<<num_of_blocks, num_of_threads_per_block>>>
	// kernel << <2, 128 >> > ();
	// kernel_dwa << <2, 128 >> > (2);
	// kernel_czy_podzielny_przez_trzy << <2, 128 >> > ();
	// kernel100_index_blok << <2, 128 >> > ();
	// kernel_index_warp_parzyste << <2, 128 >> > ();
	
	int n = 1000;
	int * a, * b, * c; // wskaźniki do tablic
	int * d_a, * d_b, * d_c; // wskaźniki na pamięć GPU
	a = new int[n];
	b = new int[n]; // alokacja pamięci do RAM
	c = new int[n];
	cudaMalloc(&d_a, n * sizeof(int));
	cudaMalloc(&d_b, n * sizeof(int)); // alokacja na GPU
	cudaMalloc(&d_c, n * sizeof(int));
	cudaMemcpy(d_a, a, n * sizeof(int));
	cudaMemcpy(d_b, a, n * sizeof(int)); // kopiowanie danych do GPU
	cudaMemcpyHostToDevice;
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize; //konfiguracja parametrów kernela

	kernel_roznica_tab << blockDim, blockSize >> > ();


	return 0;
}
int main() // function, execute on CPU
{
	// format: <<<num_of_blocks, num_of_threads_per_block>>>
	// kernel << <2, 128 >> > ();
	// kernel_dwa << <2, 128 >> > (2);
	// kernel_czy_podzielny_przez_trzy << <2, 128 >> > ();
	// kernel100_index_blok << <2, 128 >> > ();
	kernel_index_warp_parzyste << <2, 128 >> > ();
	return 0;
}
