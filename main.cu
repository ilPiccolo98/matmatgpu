#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#define MIN_RANDOM_NUMBER -1000
#define MAX_RANDOM_NUMBER 1000
#define LIMIT_DIMENSION 
#define START 1024
#define STEP 512
#define LIMIT 512 * 8

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
} 

__global__ void kernel1(double *Adev, double *Bdev, double *Cdev, int N, int M, int P)
{
    int idglob_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idglob_y = blockDim.y * blockIdx.y + threadIdx.y;
    double sum = Cdev[idglob_x * P + idglob_y];
    int k;
    for(k = 0; k != M; ++k)
        sum += Adev[idglob_x * M + k] * Bdev[idglob_y + k * P];
    Cdev[idglob_x * P + idglob_y] = sum;
}

__global__ void kernel2(double *Adev, double *Bdev, double *Cdev, int N, int M, int P)
{
    __shared__ double Ashared[32][32];
    __shared__ double Bshared[32][32];
    int idglob_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idglob_y = blockDim.y * blockIdx.y + threadIdx.y;
    double sum = Cdev[idglob_x * P + idglob_y];
    int k;
    for(k = 0; k != M / 32; ++k)
    {
        Ashared[threadIdx.x][threadIdx.y] = Adev[idglob_x * M + threadIdx.y + 32 * k];
        Bshared[threadIdx.x][threadIdx.y] = Bdev[idglob_y * M + (32 * k + threadIdx.x) * P];
        __syncthreads();
        int kk;
        for(kk = 0; kk != 32; ++kk)
            sum += Ashared[threadIdx.x][kk] * Bshared[kk][threadIdx.y];
        __syncthreads();
    }
    Cdev[idglob_x * P + idglob_y] = sum;
}

void matmatgpu1(int lda, int ldb, int ldc, double *A, double *B, double *C, int N, int M, int P)
{
    int max = 0;
    if(N > M)
        max = N;
    else
        max = M;
    if(P > max)
        max = P;
    double *buffer = (double*)malloc(sizeof(double) * max * max);
    double *Adev;
    double *Bdev;
    double *Cdev;
    cudaMalloc((void**)&Adev, sizeof(double) * N * M);
    cudaMalloc((void**)&Bdev, sizeof(double) * N * P);
    cudaMalloc((void**)&Cdev, sizeof(double) * N * P);
    int i, j;
    for(i = 0; i != N; ++i)
        for(j = 0; j != M; ++j)
            buffer[i * M + j] = A[i * lda + j];
    cudaMemcpy(Adev, buffer, sizeof(double) * N * M, cudaMemcpyHostToDevice);
    for(i = 0; i != M; ++i)
        for(j = 0; j != P; ++j)
            buffer[i * P + j] = B[i * ldb + j];
    cudaMemcpy(Bdev, buffer, sizeof(double) * N * P, cudaMemcpyHostToDevice);
    for(i = 0; i != N; ++i)
        for(j = 0; j != P; ++j)
            buffer[i * P + j] = C[i * ldc + j];
    cudaMemcpy(Cdev, buffer, sizeof(double) * N * P, cudaMemcpyHostToDevice);
    int BlockDimRow = 32;
    int BlockDimCol = 32;
    dim3 DimBlock(BlockDimRow, BlockDimCol);
    dim3 DimGrid(N / BlockDimRow, P / BlockDimCol);
    kernel2<<<DimGrid, DimBlock>>>(Adev, Bdev, Cdev, N, M, P);
    cudaDeviceSynchronize();
    cudaMemcpy(buffer, Cdev, sizeof(double) * N * P, cudaMemcpyDeviceToHost);
    for(i = 0; i != N ; ++i)
        for(j = 0; j != P; ++j)
            C[i * ldc + j] = buffer[i * P + j];
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
}

double get_random_number(double min, double max)
{
    double scale = rand() / (double) RAND_MAX;
	return min + scale * (max - (min));
    return rand() % 20;
}

double* init_matrix(int rows, int columns)
{
    double *matrix = (double*)calloc(rows * columns, sizeof(double));
    int row, column;
    for(row = 0; row != rows; ++row)
        for(column = 0; column != columns; ++column)
            matrix[row * columns + column] = get_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    return matrix;
}

void print_matrix(double *matrix, int rows, int columns)
{
    int row, column;
    for(row = 0; row != rows; ++row)
    {
        printf("Row: %d\n", row + 1);
        for(column = 0; column != columns; ++column)
            printf("%f ", matrix[row * columns + column]);
        puts("");
    }
}

int main()
{
    int i;
    for(i = START; i <= LIMIT; i += STEP)
    {
        int ldA = i;
        int N = i;
        int ldB = i;
        int M = i;
        int ldC = i;
        int P = i;
        double *A = init_matrix(N, ldA);
        double *B = init_matrix(M, ldB);
        double *C = init_matrix(P, ldC);
        long double start = get_cur_time();
        matmatgpu1(ldA, ldB, ldC, A, B, C, N, M, P);
        long double end = get_cur_time();
        long double time = end - start;
        long double gflops = 2 * pow(i, 3) / time / pow(10, 9);
        printf("Dimension: %d; Time: %Lf; Gflops: %Lf\n", i, time, gflops);
        free(A);
        free(B);
        free(C);
    }
    return 0;
}
