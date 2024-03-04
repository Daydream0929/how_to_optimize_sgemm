#include <cstdio>

#define OFFSET(row, col, ld) ((row) * (ld) + col)
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
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K,
    const int THREAD_SIZE_X,
    const int THREAD_SIZE_Y,
    const bool ENABLE_DOUBLE_BUFFER
>
__global__ void sgemm_v0(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K
)
{   
    // Block and thread index
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // the threads number in block of X and Y dimension
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // malloc shared_memory for A and B
    __shared__ As[2][BLOCK_SIZE_K][BLOCK_SIZE_M]
    __shared__ Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N]

    // malloc register for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    // malloc register for A and B
    float frag_a[THREAD_SIZE_Y]
    float frag_b[THREAD_SIZE_X]

    // register load global memory
    const int ldg_num_a = (BLOCK_SIZE_M * BLOCK_SIZE_K) / THREAD_NUM_PER_BLOCK / 4;
    const int ldg_num_b = (BLOCK_SIZE_K * BLOCK_SIZE_N) / THREAD_NUM_PER_BLOCK / 4;
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];


    // threads number in one row
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_COL = BLOCK_SIZE_N / 4;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_COL;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUN_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUN_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // positio A’row and B'col
    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[(BLOCK_SIZE_N) * bx];

    // load A from global_memory to shared_memory   
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_START * 4;
    }
    // load B from global_memory to shared_memory



}

int main(int argc, char **argv)
{

}