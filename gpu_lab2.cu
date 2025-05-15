#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "cstdlib"
#include <stdio.h>

using namespace std;

__global__ void z1_index_watku()
{
	printf("%d\n",threadIdx.x);
}

__global__ void z2_pomnoz_a(int a)
{
	int warpIdx = 0;
	warpIdx = threadIdx.x / 32;
	int wynik = warpIdx * a;
	printf("%d\n", wynik);
}

__global__ void z3_czy_podzielny_przez_trzy()
{
	if (threadIdx.x % 3 == 0)
	{
		printf("index watku: %d jest podzielny przez 3\n", threadIdx.x);
	}
}

__global__ void z4_100_suma()
{
	int a = 100;
	int wyn = a + threadIdx.x + blockIdx.x;
	printf("wynik watku %d to %d\n", threadIdx.x, wyn);
}

__global__ void z5_czy_suma_indexow_parzysta()
{
	int warpIdx = threadIdx.x / 32;
	int allidx = threadIdx.x + warpIdx + blockIdx.x;
	if (allidx % 2 == 0)
	{
		printf("%d suma indexu watku, bloku i warpa to %d i jest parzysta\n", threadIdx.x, allidx);
	}
}

#define N 1024
#define BLOCK_SIZE 256

__global__ void z6_roznica(float* a, float* b, float* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		c[idx] = a[idx] - b[idx];
	}
}

void z6()
{
	float* cpu_a = new float[N];
	float* cpu_b = new float[N];
	float* cpu_c = new float[N];
	float* gpu_a, * gpu_b, * gpu_c;

	for (int i = 0;i < N;i++)
	{
		cpu_a[i] = static_cast<float>(i+10);
		cpu_b[i] = static_cast<float>(i);
	}

	cudaMalloc(&gpu_a, N * sizeof(float));
	cudaMalloc(&gpu_b, N * sizeof(float));
	cudaMalloc(&gpu_c, N * sizeof(float));

	cudaMemcpy(gpu_a, cpu_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, N * sizeof(float), cudaMemcpyHostToDevice);

	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	z6_roznica <<< gridsize, BLOCK_SIZE >>> (gpu_a, gpu_b, gpu_c);
	cudaDeviceSynchronize();

	cudaMemcpy(cpu_c, gpu_c, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0;i < N;i++)
	{
		printf("%f\n", cpu_c[i]);
	}

	delete[]cpu_a;
	delete[]cpu_b;
	delete[]cpu_c;
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
}

__global__ void z7_suma_roznica(float* a, float* b, float* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		if (idx % 2 == 0)
			c[idx] = a[idx] + b[idx];
		else
			c[idx] = a[idx] - b[idx];
	}
}

void z7()
{
	float* cpu_a = new float[N];
	float* cpu_b = new float[N];
	float* cpu_c = new float[N];
	float* gpu_a, * gpu_b, * gpu_c;

	for (int i = 0;i < N;i++)
	{
		cpu_a[i] = static_cast<float>(i + 10);
		cpu_b[i] = static_cast<float>(i);
	}

	cudaMalloc(&gpu_a, N * sizeof(float));
	cudaMalloc(&gpu_b, N * sizeof(float));
	cudaMalloc(&gpu_c, N * sizeof(float));

	cudaMemcpy(gpu_a, cpu_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, N * sizeof(float), cudaMemcpyHostToDevice);

	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	z7_suma_roznica << <gridsize, BLOCK_SIZE >> > (gpu_a, gpu_b, gpu_c);

	cudaMemcpy(cpu_c, gpu_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0;i < N;i++)
	{
		printf("%f\n", cpu_c[i]);
	}

	free(cpu_a);
	free(cpu_b);
	free(cpu_c);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
}

__global__ void z8_czy_warp_parzyste(float* a, float* b, float* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warpIdx = threadIdx.x / 32;
	if (idx < N)
	{
		if (warpIdx % 2 == 0)
			c[idx] = a[idx] + b[idx];
		else
			c[idx] = a[idx] - b[idx];
	}
}

void z8()
{
	float* cpu_a = new float[N];
	float* cpu_b = new float[N];
	float* cpu_c = new float[N];
	float* gpu_a, * gpu_b, * gpu_c;

	for (int i = 0;i < N;i++)
	{
		cpu_a[i] = static_cast<float>(i + 10);
		cpu_b[i] = static_cast<float>(i);
	}

	cudaMalloc(&gpu_a, N * sizeof(float));
	cudaMalloc(&gpu_b, N * sizeof(float));
	cudaMalloc(&gpu_c, N * sizeof(float));

	cudaMemcpy(gpu_a, cpu_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, N * sizeof(float), cudaMemcpyHostToDevice);

	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	z8_czy_warp_parzyste << <gridsize, BLOCK_SIZE >> > (gpu_a, gpu_b, gpu_c);

	cudaMemcpy(cpu_c, gpu_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0;i < N;i++)
	{
		printf("%f\n", cpu_c[i]);
	}

	free(cpu_a);
	free(cpu_b);
	free(cpu_c);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
}

__global__ void z9_suma_duzych_wektorow(float* a, float* b, float* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		c[idx] = a[idx] + b[idx];
	}
}

void z9()
{
	int super_duper_duze = 16 * 1024 * 1024 * 1024/ sizeof(float); //16GB, na potrzeby sprawdzenie usuń dwa 1024
	float* a = new float[super_duper_duze];
	float* b = new float[super_duper_duze];
	float* af = new float[N];
	float* bf = new float[N];
	float* c = new float[N];
	float* da, * db, * dc;

	for (int i = 0; i < super_duper_duze; i++)
	{
		a[i] = static_cast<float>(i);
		b[i] = static_cast<float>(i);
	}

	cudaMalloc(&da, N * sizeof(float));
	cudaMalloc(&db, N * sizeof(float));
	cudaMalloc(&dc, N * sizeof(float));

	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	for (int i = 0; i < super_duper_duze / N;i++)
	{
		for (int j = 0; j < N; j++)
		{
			af[j] = a[i * N + j];
			bf[j] = b[i * N + j];
		}
		cudaMemcpy(da, af, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(db, bf, N * sizeof(float), cudaMemcpyHostToDevice);

		z9_suma_duzych_wektorow << < gridsize, BLOCK_SIZE >> > (da, db, dc);


		cudaMemcpy(c, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

		for (int z = 0;z < N;z++)
		{
			printf("%f\n", c[z]);
		}
	}
	free(a);
	free(af);
	free(b);
	free(bf);
	free(c);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

int main()
{
	//z1_index_watku << <2, 128 >> > ();
	//z2_pomnoz_a << <2, 128 >> > (2);
	//z3_czy_podzielny_przez_trzy << <2, 128 >> > ();
	//z4_100_suma << <2, 128 >> > ();
	//z5_czy_suma_indexow_parzysta << <2, 128 >> > ();
	//z6();
	//z7();
	//z8();
	//z9();
	return 0;
}
