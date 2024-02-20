# how_to_optimize_sgemm

## Setup
```
nvcc -o gemm.out gemm.cu -lcublas 
``` 

## M = K = N 4096   
* Matrix0                   Time = 120ms
* Matrix1                   Time = 80ms
* cublas                    Time = 20ms
* nvidia_Matrix             Time = 120 ms.
* nvidia_Matrix_shared      Time = 80 ms.