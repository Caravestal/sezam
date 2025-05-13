#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

#define N 1024
#define BLOCK_SIZE 256 //blockDim.x liczba watkow w bloku

__global__ void z1_kernel_mnog_skalar(float skalar, float* wejscie_tab, float* wyjscie_tab)
{
	__shared__ float sdata[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		sdata[threadIdx.x] = wejscie_tab[idx];
		__syncthreads();
		sdata[threadIdx.x] *= skalar;
		wyjscie_tab[idx] = sdata[threadIdx.x];
	}
}

__global__ void z2_dod_tab(float* a, float* b, float* wyjscie_tab)
{
	__shared__ float adata[BLOCK_SIZE];
	__shared__ float bdata[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		adata[threadIdx.x] = a[idx];
		bdata[threadIdx.x] = b[idx];
		__syncthreads();
		wyjscie_tab[idx] = adata[threadIdx.x] + bdata[threadIdx.x];
	}

}

__global__ void z3_kernel_kopi(float* a, float* b)
{
	__shared__ float adata[BLOCK_SIZE];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		adata[threadIdx.x] = a[idx];
		__syncthreads();
		b[N - idx] = adata[threadIdx.x];
	}
}

__global__ void z4_kernel_warp_shuffle(float* wejscie_tab, int n)
{
	// pamiec wspoldzielona do trzymania sum z warp'ow, rozmiar elastyczny
	extern __shared__ float shared[];
	int tid = threadIdx.x;
	int index = 2 * blockIdx.x * blockDim.x + tid;
	float sum = 0.0f; // wartosc do przechowywania w rejestrze
	sum = (index < n ? wejscie_tab[index] : 0.0f) + (index + blockDim.x < n ? wejscie_tab[index + blockDim.x] : 0.0f);

	// petla sumujaca wartosci z rejestrow tego warpu
		// offset to kolejno 16, 8, ..., 1
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		// sum = obecna sum z tego watku + wartosc watku od ID 
		// offset w tym warp'ie
		sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
	}

	// jezeli biezacy id watku to pierwszy w warp'ie
	if (tid % warpSize == 0) {
		// ustaw wartosc sumy tego warp'u w pamieci wspoldzielonej 
		// na jego pozycji
		shared[tid / warpSize] = sum;
	}
	__syncthreads();

	// jezeli id obecnego watku miesci sie w warpie, to uzyj go do wykonania akcji
	if (tid < warpSize) {
		// jezeli id watku odpowiada pozycji wartosci w tablicy shared
		sum = (tid < (blockDim.x / warpSize)) ? shared[tid] : 0.0f;
		// offset po kolei 16, 8, ..., 1
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			// zredukuj sumy w warpie, zawarte w rejestrach, 
			// to sa sumy czastkowe ze wszystkich warp'ow z bloku
			sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
		}
	}

	if (tid == 0) { // jezeli pierwszy watek w bloku
		wejscie_tab[blockIdx.x] = sum; // ustaw sume wszystkich warp'ow
	}
}

void z1()
{
	float* wejscie_cpu = new float[N];
	float* wyjscie_cpu = new float[N];
	float* wejscie_gpu, * wyjscie_gpu;
	float skalar = 2.5f;

	for (int i = 0; i < N; i++)
	{
		wejscie_cpu[i] = static_cast<float>(i); // zmiana i na float (wpisuje w wejscie i)
	}

	cudaMalloc(&wejscie_gpu, N * sizeof(float));
	cudaMalloc(&wyjscie_gpu, N * sizeof(float));

	cudaMemcpy(wejscie_gpu, wejscie_cpu, N * sizeof(float), cudaMemcpyHostToDevice); // z hosta(cpu) do urzadzenie(gpu) 

	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // liczy ile bloków potrzeba żeby pokryć wszystkie N: 1024/256 = 4 (4bloki po 256 wątków)

	z1_kernel_mnog_skalar << < gridsize, BLOCK_SIZE >> > (skalar, wejscie_gpu, wyjscie_gpu);

	cudaMemcpy(wyjscie_cpu, wyjscie_gpu, N * sizeof(float), cudaMemcpyDeviceToHost); // przepisanie z gpu na cpu zeby dało sie to odczytac

	for (int i = 0; i < N; ++i) {
		cout << wyjscie_cpu[i] << " ";
	}
	cout << endl;

	delete[] wejscie_cpu;
	delete[] wyjscie_cpu;
	cudaFree(wejscie_gpu);
	cudaFree(wyjscie_gpu);
}

void z2()
{
	float* cpu_a = new float[N];
	float* cpu_b = new float[N];
	float*cpu_wyjscie = new float[N];
	float* gpu_a, * gpu_b, * gpu_wyjscie;

	for (int i = 0; i < N; i++)
	{
		cpu_a[i] = static_cast<float>(i);
		cpu_b[i] = static_cast<float>(i*10);
	}

	cudaMalloc(&gpu_a, N * sizeof(float));
	cudaMalloc(&gpu_b, N * sizeof(float));
	cudaMalloc(&gpu_wyjscie, N * sizeof(float));

	cudaMemcpy(gpu_a, cpu_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, cpu_b, N * sizeof(float), cudaMemcpyHostToDevice);

	int gridsize = N + BLOCK_SIZE - 1 / BLOCK_SIZE; // można jeszcze gridsize liczyć w N/BLOCK_SIZE : używamy tego gdy dzieli się równo przez blocksize, obecna wersja jest bardziej ogólna i obejmuje również wersje gdy n jest niepodzielne przez blocksize
	
	z2_dod_tab << < gridsize, BLOCK_SIZE >> > (gpu_a, gpu_b, gpu_wyjscie);

	cudaMemcpy(cpu_wyjscie, gpu_wyjscie, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N; i++)
	{
		cout << cpu_wyjscie[i] << " ";
	}
	cout << endl;

	delete[] cpu_a;
	delete[] cpu_b;
	delete[] cpu_wyjscie;
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_wyjscie);
}

void z3()
{
	float* cpu_a = new float[N];
	float* cpu_b = new float[N];
	float* gpu_a,*  gpu_b;

	for (int i = 0; i < N; i++)
	{
		cpu_a[i] = static_cast<float>(i);
	}

	cudaMalloc(&gpu_a, N * sizeof(float));
	cudaMalloc(&gpu_b, N * sizeof(float));

	cudaMemcpy(gpu_a, cpu_a, N * sizeof(float), cudaMemcpyHostToDevice);

	int gridsize = N + BLOCK_SIZE - 1 / BLOCK_SIZE;

	z3_kernel_kopi << < gridsize, BLOCK_SIZE >> > (gpu_a, gpu_b);

	cudaMemcpy(cpu_b, gpu_b, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		cout << cpu_b[i] << " ";
	}
	cout << endl;

	delete[] cpu_a;
	delete[] cpu_b;
	cudaFree(gpu_a);
	cudaFree(gpu_b);
}

float cpuRedukcja(float* wejscie, int n) {
	float sum = 0.0f;
	for (int i = 0; i < n; ++i) {
		sum += wejscie[i];
	}
	return sum;
}

void z4()
{
	size_t bytes = N * sizeof(float);

	float* cpu_wejscie = new float[N];
	float* gpu_wejscie;

	for (int i = 0; i < N; i++) 
	{
		cpu_wejscie[i] = static_cast<float>(i + 1);
	}

	cudaMalloc(&gpu_wejscie, bytes);

	cudaMemcpy(gpu_wejscie, cpu_wejscie, bytes, cudaMemcpyHostToDevice);

	int grid_size = (N +BLOCK_SIZE - 1) / BLOCK_SIZE;
	// liczba elementow zgodna z liczba warp'ow w bloku
	size_t shared_mem_size = (BLOCK_SIZE / 32) * sizeof(float);

	float total_sum = cpuRedukcja(cpu_wejscie, N);
	cout << "Mean (CPU): " << total_sum / N << endl;

	while (grid_size > 1) {
		z4_kernel_warp_shuffle << <grid_size, BLOCK_SIZE, shared_mem_size >> > (gpu_wejscie, N);
		cudaDeviceSynchronize();

		
		grid_size = N + BLOCK_SIZE - 1 / BLOCK_SIZE;
	}

	z4_kernel_warp_shuffle << <1, BLOCK_SIZE, shared_mem_size >> > (gpu_wejscie, N);
	cudaDeviceSynchronize();

	cudaMemcpy(cpu_wejscie, gpu_wejscie, sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Mean (GPU): " << cpu_wejscie[0] / N << endl;

	cudaFree(gpu_wejscie);
	delete[] cpu_wejscie;
}

int main()
{
	//z1();
	//z2();
	//z3();
	z4();

	return 0;
}
