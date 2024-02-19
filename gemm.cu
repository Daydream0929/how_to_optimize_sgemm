#include <iostream>
#include "error.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 行主序
#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

// A[M][K] B[K][N] C[M][N]
const int M = 2048;
const int K = 2048;
const int N = 2048;

const int ITERATION = 50;

// block_size and grid_size
const dim3 threads_per_block = {16, 16, 1};
const dim3 blocks_per_grid = {M / threads_per_block.x, N / threads_per_block.y, 1};


// kernel 行主序
__global__ void matrixMul0(const float *a, const float *b, float *c, int M, int N, int K)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int lda = K, ldb = N, ldc = N;
    if (col < N && row < M)
    {
        float value = 0;
        for (int k = 0; k < K; k++) 
        {
            value += A(row, k) * B(k, col);
        }
        C(row, col) = value;
    }
}

void test_matrixMat0()
{
    // allocate host for A, B, C
    float *h_a = (float *)malloc(sizeof(float) * M * K);
    float *h_b = (float *)malloc(sizeof(float) * K * N);
    float *h_c = (float *)malloc(sizeof(float) * M * N);
    float *h_c1 = (float *)malloc(sizeof(float) * M * N); // cublas

    for (int i = 0; i < M * K; i++)
        h_a[i] = i % 7;
    for (int i = 0; i < K * N; i++)
        h_b[i] = i % 7;

    // test device_c result
    float *test_c = (float *)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            float value = 0;
            for (int k = 0; k < K; k ++) {
                //test_c[i][j] += h_a[i][k] * h_b[k][j]
                value += h_a[i * K + k] * h_b[k * N + j];
            }
            test_c[i * N + j] += value;
        }
    }

    // allocate device for A, B, C
    float *d_a;
    CHECK(cudaMalloc((void **)&d_a, sizeof(float) * M * K));
    float *d_b;
    CHECK(cudaMalloc((void **)&d_b, sizeof(float) * K * N));
    float *d_c;
    CHECK(cudaMalloc((void **)&d_c, sizeof(float) * M * N));

    // copy h_a h_b for d_a d_b
    CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    // record the MatrixMul0 time
    float t_sum = 0, t2_sum = 0;
    for (int repeat = 0; repeat <= ITERATION; repeat++)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // MatrixMul0 kernel
        matrixMul0<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, M, N, K);

        CHECK(cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost)); // 隐式同步

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        } 
        else
        {
            for (int i = 0; i < M * N; i ++) {
                if (abs(h_c[i] - test_c[i]) > 1e-5) {
                    printf("MatrixMul0 Result Error!\n");
                    break;
                }
            }
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / ITERATION;
    const float t_err = sqrt(t2_sum / ITERATION - t_ave * t_ave);
    printf(" ----- MatrixMul0 ----- \n");
    printf("Time = %g +- %g ms.\n\n", t_ave, t_err);


    // copy h_a h_b for d_a d_b
    CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    // record the cublas time
    t_sum = 0, t2_sum = 0;
    for (int repeat = 0; repeat < ITERATION; repeat ++) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // cublas kernel
        cublasHandle_t blas_handle;
        cublasCreate(&blas_handle);
        float alpha = 1.0f, beta = 0.0f;
        // CHECK(cudaMemcpy(d_c, h_c, sizeof(float) * M * N, cudaMemcpyHostToDevice));
        cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_a, K, d_b, N, &beta, d_c, N);

        CHECK(cudaMemcpy(h_c1, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost)); // 隐式同步

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        } 
        else
        {
            for (int i = 0; i < M * N; i ++) {
                if (abs(h_c1[i] - test_c[i]) > 1e-5) {
                    printf("cublas Result Error!\n");
                    break;
                }
            }
            // for (int i = 0; i < M * N; i ++) {
            //     std::cout << i << " h_c1 " << h_c1[i] << " test_c " << test_c[i] << std::endl;
            // }
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

    }

    const float t_ave2 = t_sum / ITERATION;
    const float t_err2 = sqrt(t2_sum / ITERATION - t_ave * t_ave);
    printf(" ----- cublas ----- \n");
    printf("Time = %g +- %g ms.\n\n", t_ave2, t_err2);

    // Free Memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c1);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}



int main()
{
    test_matrixMat0();

    return 0;
}