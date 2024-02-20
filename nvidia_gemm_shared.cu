#include "error.cuh"

const int M = 1024 * 4;
const int K = 1024 * 4;
const int N = 1024 * 4;

const int ITERATION = 10;

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct 
{
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride; 
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device_memory
    Matrix d_A;
    d_A.width   = d_A.stride = A.width;
    d_A.height  = A.height;
    size_t size = A.width * A.height * sizeof(float);
    CHECK(cudaMalloc(&d_A.elements, size));
    CHECK(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

    Matrix d_B;
    d_B.width   = d_B.stride = B.width;
    d_B.height  = B.height;
    size = B.width * B.height * sizeof(float);
    CHECK(cudaMalloc(&d_B.elements, size));
    CHECK(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

    // Allocate C in device memory
    Matrix d_C;
    d_C.width   = d_C.stride = C.width;
    d_C.height  = C.height;
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
    dim3 blocks_per_grid(B.width / threads_per_block.x, A.height / threads_per_block.y);


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

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{   
    // Block row and column
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, block_row, block_col);

    // Each thread computes one element of Csub by accumulating results into Cvalue
    float value = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); m ++) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, block_row, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, block_col);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; e ++) {
            value += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    SetElement(Csub, row, col, value);
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

