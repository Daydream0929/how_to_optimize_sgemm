#include <iostream>
#include "error.cuh"

// A[M][K] B[K][N] C[M][N]
const int M = 1024 * 8;
const int K = 1024 * 8;
const int N = 1024 * 8;

const int ITERATION = 50;

const dim3 threads_per_block = {128};
const dim3 blocks_per_grid = {128};

// 行主序
__global__ void matrixMul0(const float *A, const float *B, float *C, int M, int N, int K)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < M && ty < N)
    {
        float c = 0;
        for (int i = 0; i < K; i++)
        {
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

void test_matrixMat0()
{
    // allocate host for A, B, C
    float *h_a = (float *)malloc(sizeof(float) * M * K);
    float *h_b = (float *)malloc(sizeof(float) * K * N);
    float *h_c = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++)
        h_a[i] = i % 17;
    for (int i = 0; i < K * N; i++)
        h_b[i] = i % 17;

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

    // record the time
    float t_sum = 0, t2_sum = 0;
    for (int repeat = 0; repeat <= ITERATION; repeat++)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        // kernel
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

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / ITERATION;
    const float t_err = sqrt(t2_sum / ITERATION - t_ave * t_ave);
    printf("Time = %g +- %g ms.\n", t_ave, t_err);

    // Free Memory
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}

int main()
{
    test_matrixMat0();

    return 0;
}