#include "error.cuh"

const int M = 1024 * 4;
const int K = 1024 * 4;
const int N = 1024 * 4;

const int ITERATION = 10;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct 
{
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device_memory
    Matrix d_A;
    d_A.width = A.width, d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    CHECK(cudaMalloc(&d_A.elements, size));
    CHECK(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

    Matrix d_B;
    d_B.width = B.width, d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    CHECK(cudaMalloc(&d_B.elements, size));
    CHECK(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width, d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    CHECK(cudaMalloc(&d_C.elements, size));

    // test device_c result
    float *test_c = (float *)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            float value = 0;
            for (int k = 0; k < K; k ++) {
                //test_c[i][j] += h_a[i][k] * h_b[k][j]
                value += A.elements[i * K + k] * B.elements[k * N + j];
            }
            test_c[i * N + j] += value;
        }
    }

    // Invoke kernel
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(B.width / BLOCK_SIZE, A.height / BLOCK_SIZE);

    // Record the assuming time
    float t_sum = 0, t2_sum = 0;
    for (int repeat = 0; repeat < ITERATION; repeat ++) 
    {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        MatMulKernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C);

        // Read C from device memory
        CHECK(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));

        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
        printf("Time = %g ms .\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        } 
        else
        {
            for (int i = 0; i < M * N; i ++) {
                if (abs(C.elements[i] - test_c[i]) > 1e-5) {
                    printf("MatrixMul Result Error!\n");
                    break;
                }
            }
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));

    }

    const float t_ave = t_sum / ITERATION;
    const float t_err = sqrt(t2_sum / ITERATION - t_ave * t_ave);
    printf(" ----- MatrixMul ----- \n");
    printf("Time = %g +- %g ms.\n\n", t_ave, t_err);


    // Free device memory
    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(test_c);
    CHECK(cudaFree(d_A.elements));
    CHECK(cudaFree(d_B.elements));
    CHECK(cudaFree(d_C.elements));
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
    float value = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    for (int e = 0; e < A.width; e ++) {
        value += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
    C.elements[row * C.width + col] = value;    
}

int main()
{
    // Allocate A B C
    Matrix A, B, C;
    A.height = M, A.width = K;
    B.height = K, B.width = N;
    C.height = M, C.width = N;

    A.elements = (float *)malloc(sizeof(float) * M * K);
    B.elements = (float *)malloc(sizeof(float) * K * N);
    C.elements = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i ++) A.elements[i] = i % 7;
    for (int i = 0; i < K * N; i ++) B.elements[i] = i % 7;

    MatMul(A, B, C);

    return 0;
}

