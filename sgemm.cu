/*
github: https://github.com/nicolaswilde/cuda-sgemm/blob/main/sgemm.cu
zhihu: https://zhuanlan.zhihu.com/p/518857175
*/

#include "error.cuh"
#include <cublas_v2.h>
#include <float.h>

// row-major
#define OFFSET(row, col, ld) ((row)* (ld) + (col))

// cpu navce version
void cpu_sgemm(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m ++) {
        for (int n = 0; n < N; n ++) {
            float value = 0;
            for (int k = 0; k < K; k ++) {
                value += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] += value;
        }
    }
}

// gpu naive version
__global__ void naive_sgemm(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c, const int M, const int N, const int K)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int m = blockDim.y * blockIdx.y + threadIdx.y;
    if (m < M && n < n) {
        float value = 0.0f;
        for (int k = 0; k < K; k ++) {
            value += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];   
        }
        c[OFFSET(m, n, N)] += value;
    }
}

float test_performance(void (*gpu_sgemm)(float *, float *, float *, const int, const int, const int), 
                        dim3 grid_size, dim3 block_size, 
                        const int M, const int N, const int K,
                        const int repeat)
{
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i ++) {
        gpu_sgemm<<<grid_size, block_size>>>(d_a, d_b, d_c, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

float test_max_error(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpu_sgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float test_cublas_max_error(const int M, const int N, const int K)
{
    
}

float test_cublas_cerformance(const int M, const int N, const int K, const int repeat)
{
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0f;
    float cublas_beta = 0.0f;


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i ++) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, N);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

int main()
{
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    // const int K_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int outer_repeat = 10, inner_repeat = 1;

    {
        printf("\nKernal = cublas\n");
        {
            const int M = 512, N = 512, K = 512;
            float max_error = test_cublas_max_error(M, N, K);
            printf("Max Error = %f\n", max_error);
        }
    }

    {
        const int TESTNUM = 15;

        for (int i = 0; i < TESTNUM; i++) {
            const int M = M_list[i], N = N_list[i], K = K_list[i];

            double max_sec = 0.0;
            double min_sec = DBL_MAX;
            double total_sec = 0.0;

            for (int j = 0; j < outer_repeat; j++) {
                double this_sec = test_cublas_cerformance(M, N, K, inner_repeat);
                max_sec = max(max_sec, this_sec);
                min_sec = min(min_sec, this_sec);
                total_sec += this_sec;
            }

            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

            printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
        }
    }

    {
        printf("\nKernal = naiveSgemm\n");

        const int BM = 32, BN = 32;
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = naive_sgemm;

        {
            const int M = 512, N = 512, K = 512;
            dim3 blockDim(BN, BM);
            dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
            float max_error = test_max_error(gpuSgemm, gridDim, blockDim, M, N, K);
            printf("Max Error = %f\n", max_error);
        }

        {
            const int TESTNUM = 15;

            for (int i = 0; i < TESTNUM; i++) {
                const int M = M_list[i], N = N_list[i], K = K_list[i];

                dim3 blockDim(BN, BM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = test_performance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }


    return 0;
}