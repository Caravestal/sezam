#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#define N2 10

using namespace std;


__global__ void z1a_gpu(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int index = row * N + col;

        C[index] = A[index] * B[index];
    }
}

__global__ void z1b_gpu(float* A, float* B, float* C)
{
    __shared__ float sA[N2 * N2];
    __shared__ float sB[N2 * N2];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * N2 + col;

    if (row < N2 && col < N2)
    {
        sA[index] = A[index];
        sB[index] = B[index];
    }

    __syncthreads();


    if (row < N2 && col < N2)
    {
        C[index] = sA[index] * sB[index];
    }
}

__device__ float standardDeviation_sum = 0;
__device__ float standardDeviation = 0;
__device__ float sum = 0;
__device__ float mean = 0;

__global__ void z2a_gpu(float* A, float* B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * N2 + col;

    if (row < N2 && col < N2)
    {
        sum += A[index];
    }

    __syncthreads();

    mean = sum / N2;

    __syncthreads();

    if(row < N2 && col < N2)
    {
        standardDeviation_sum += pow(A[index] - mean, 2);
    }
    
    __syncthreads();

    standardDeviation = sqrt(standardDeviation_sum / N2);

    __syncthreads();

    B[index] = (A[index] - mean) / standardDeviation;

}


__global__ void z2b_gpu(float* A)
{
    __shared__ float standardDeviation_sum[1];
    __shared__ float standardDeviation[1];
    __shared__ float sum[1];
    __shared__ float mean[1];

    __shared__ float sharedA[N2 * N2];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * N2 + col;

    if (row < N2 && col < N2)
    {
        sharedA[index] = A[index];
    }

    __syncthreads();

    if (row < N2 && col < N2)
    {
        sum[1] += sharedA[index];
    }

    __syncthreads();

    mean[1] = sum[1] / N2;

    __syncthreads();

    if (row < N2 && col < N2)
    {
        standardDeviation_sum[1] += pow(sharedA[index] - mean[1], 2);
    }

    __syncthreads();

    standardDeviation[1] = sqrt(standardDeviation_sum[1] / N2);

    __syncthreads();

    A[index] = (sharedA[index] - mean[1]) / standardDeviation[1];

}

void display_matrix(const float* matrix, int N, string title = "") {

    if (title != "") {
        cout << title << endl;
    }

    cout << "-------------------" << endl;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void initialize_matrix(float matrix[], int N, bool withZero = false) {
    for (int i = 0; i < N * N; i++)
    {
        if (withZero) {
            matrix[i] = 0;
        }
        else {
            matrix[i] = rand() % 10;
        }
    }
}

void z1a()
{

    srand(std::time(nullptr));
    const int N = 10;

    float* firstMatrix = new float[N * N];
    float* secondMatrix = new float[N * N];
    float* outputMatrix = new float[N * N];

    float* firstMatrix_gpu, * secondMatrix_gpu, * outputMatrix_gpu;
    cudaMalloc((void**)&firstMatrix_gpu, N * N * sizeof(float));
    cudaMalloc((void**)&secondMatrix_gpu, N * N * sizeof(float));
    cudaMalloc((void**)&outputMatrix_gpu, N * N * sizeof(float));

    initialize_matrix(firstMatrix, N);
    initialize_matrix(secondMatrix, N);
    initialize_matrix(outputMatrix, N, true);
    display_matrix(firstMatrix, N, "First matrix");
    display_matrix(secondMatrix, N, "Second matrix");


    cudaMemcpy(firstMatrix_gpu, firstMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(secondMatrix_gpu, secondMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outputMatrix_gpu, outputMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    z1a_gpu << <numBlocks, threadsperblock >> > (firstMatrix_gpu, secondMatrix_gpu, outputMatrix_gpu, N);

    cudaMemcpy(outputMatrix, outputMatrix_gpu, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    display_matrix(outputMatrix, N, "Output matrix");
    delete[] firstMatrix;
    delete[] secondMatrix;
    delete[] outputMatrix;

    cudaFree(firstMatrix_gpu);
    cudaFree(secondMatrix_gpu);
    cudaFree(outputMatrix_gpu);

}

void z1b()
{
    srand(std::time(nullptr));

    float* firstMatrix = new float[N2 * N2];
    float* secondMatrix = new float[N2 * N2];
    float* outputMatrix = new float[N2 * N2];

    float* firstMatrix_gpu, * secondMatrix_gpu, * outputMatrix_gpu;
    cudaMalloc((void**)&firstMatrix_gpu, N2 * N2 * sizeof(float));
    cudaMalloc((void**)&secondMatrix_gpu, N2 * N2 * sizeof(float));
    cudaMalloc((void**)&outputMatrix_gpu, N2 * N2 * sizeof(float));

    initialize_matrix(firstMatrix, N2);
    initialize_matrix(secondMatrix, N2);
    initialize_matrix(outputMatrix, N2, true);
    display_matrix(firstMatrix, N2, "First matrix");
    display_matrix(secondMatrix, N2, "Second matrix");


    cudaMemcpy(firstMatrix_gpu, firstMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(secondMatrix_gpu, secondMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outputMatrix_gpu, outputMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N2 + block_size - 1) / block_size, (N2 + block_size - 1) / block_size);

    z1b_gpu << <numBlocks, threadsperblock >> > (firstMatrix_gpu, secondMatrix_gpu, outputMatrix_gpu);

    cudaMemcpy(outputMatrix, outputMatrix_gpu, N2 * N2 * sizeof(float), cudaMemcpyDeviceToHost);
    display_matrix(outputMatrix, N2, "Output matrix");
    delete[] firstMatrix;
    delete[] secondMatrix;
    delete[] outputMatrix;

    cudaFree(firstMatrix_gpu);
    cudaFree(secondMatrix_gpu);
    cudaFree(outputMatrix_gpu);

}

void z2a()
{
    srand(std::time(nullptr));

    float* firstMatrix = new float[N2 * N2];
    float* outputMatrix = new float[N2 * N2];

    float* firstMatrix_gpu, * outputMatrix_gpu;
    cudaMalloc((void**)&firstMatrix_gpu, N2 * N2 * sizeof(float));
    cudaMalloc((void**)&outputMatrix_gpu, N2 * N2 * sizeof(float));

    initialize_matrix(firstMatrix, N2);
    initialize_matrix(outputMatrix, N2, true);
    display_matrix(firstMatrix, N2, "First matrix");


    cudaMemcpy(firstMatrix_gpu, firstMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(outputMatrix_gpu, outputMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(10, 10);

    dim3 numBlocks(1, 1);

    z2a_gpu << <numBlocks, threadsperblock >> > (firstMatrix_gpu, outputMatrix_gpu);

    cudaMemcpy(outputMatrix, outputMatrix_gpu, N2 * N2 * sizeof(float), cudaMemcpyDeviceToHost);
    display_matrix(outputMatrix, N2, "Output matrix");
    delete[] firstMatrix;
    delete[] outputMatrix;

    cudaFree(firstMatrix_gpu);
    cudaFree(outputMatrix_gpu);

}

void z2b()
{
    srand(std::time(nullptr));

    float* firstMatrix = new float[N2 * N2];

    float* firstMatrix_gpu, * outputMatrix_gpu;
    cudaMalloc((void**)&firstMatrix_gpu, N2 * N2 * sizeof(float));

    initialize_matrix(firstMatrix, N2);
    display_matrix(firstMatrix, N2, "First matrix");


    cudaMemcpy(firstMatrix_gpu, firstMatrix, N2 * N2 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(10, 10);

    dim3 numBlocks(1, 1);

    z2b_gpu << <numBlocks, threadsperblock >> > (firstMatrix_gpu);

    cudaMemcpy(firstMatrix, firstMatrix_gpu, N2 * N2 * sizeof(float), cudaMemcpyDeviceToHost);
    display_matrix(firstMatrix, N2, "Output matrix");
    delete[] firstMatrix;

    cudaFree(firstMatrix_gpu);
}


int main()
{
    z1a();
    //z1b();
    //z2a();
    //z2b();
    return 0;
}

