/*
github: https://github.com/Liu-xiandong/How_to_optimize_in_GPU/blob/master/sgemm/sgemm_v1.cu
zhihu: https://zhuanlan.zhihu.com/p/435908830
*/ 

#include <cstdio>
#include "assert.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0]) 

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void Sgemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
    )
{   
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // the threads nunmber in block of X and Y dimension
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;            // ???? blockDim.x
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;            // ???? blockDim.y
    const int THREAD_NUN_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;   // ???? blockDim.x * blockDim.y

    // thread_id in cur block
    const int tid = ty * THREAD_X_PER_BLOCK + tx; 
    
    // shared_memory for A and B
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];         // double * transformer A
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];         // double *             B
    
    // register_memory for C 
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    // register_memory for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    // register_memory load global memory
    const int ldg_num_a = (BLOCK_SIZE_M * BLOCK_SIZE_K) / (THREAD_NUN_PER_BLOCK * 4);
    const int ldg_num_b = (BLOCK_SIZE_K * BLOCK_SIZE_N) / (THREAD_NUN_PER_BLOCK * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_COL = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this threa
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_COL;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUN_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUN_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // position A's row and B's col
    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[BLOCK_SIZE_N * bx];

    // transfer first tile from global_memory to shared_memory
    // load A from global_memory to shared_memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_START * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(BLOCK_SIZE_M * by + A_TILE_ROW_START + i, A_TILE_COL, K)]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]     = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 1];
        As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 2];
        As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index + 3];
    }
    // load B from global memory to shared_memory
    #pragma unroll  
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL, N)]);
    }
    __syncthreads();

    // load A from shared_memory to register
    #pragma unroll
    for (int thread_y = 0;  thread_y < THREAD_SIZE_X; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }

    // load B from shared_memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_Y; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K; 

        // load next tile from global memory
        if (tile_idx < K) {
            #pragma unroll

        }

        int load_stage_idx = write_stage_idx ^= 1;

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; j ++) {

        }

        if (tile_idx < K) {
            #pragma unroll
        }

        // load first tile from shared_memory to register of next iter
        // load A from shred_memory to register
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {

        }
        // load B from shared_momory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {

        }

        // compute last tile mma THREAD_X x THREAD_Y
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y ++) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x ++) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }

    } while (tile_idx < K);

    // store back to C
    #pragma
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y ++) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x ++) {
            FETCH_FLOAT4(C[OFFSET(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y, BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x, N)])
                = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }

}

int main(int argc, char** argv)
{   
    if (argc != 4) {
        printf("usage: ./sgemm_v1.out [M] [K] [N]\n");
        return 0;
    }

    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert(M % 8 == 0);
    assert(K % 8 == 0);
    assert(N % 8 == 0);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;

    float* h_A  = (float*)malloc(bytes_A);
    float* h_B  = (float*)malloc(bytes_B);
    float* h_C  = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A, d_B, d_C;
    CHECK(cudaMalloc((void**)&d_A, bytes_A));
    CHECK(cudaMalloc((void**)&d_B, bytes_B));
    CHECK(cudaMalloc((void**)&d_C, bytes_C));

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M  = 128;
    const int BLOCK_SIZE_K  = 128;
    const int BLOCK_SIZE_N  = 128;
    const int THREAD_SIZE_X = 128;
    const int THREAD_SIZE_Y  = 128;
    const int BLOCK_SIZE_M  = 128;

}