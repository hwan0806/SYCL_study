/*
    Rule #1 - [gridDim][blockDim]
        - [블락개수][스레드개수] - [blockIdx][threadIdx]

    11.6 - C++20 지원되지 않구요, C++17로 코딩합니다
*/

#pragma nv_diag_suppress 177

#include <tpf_output.hpp>
#include <tpf_chrono_random.hpp>
#include <cuda.h>

namespace tcr = tpf::chrono_random;
tpf::sstream stream;
auto& endl = tpf::endl;

template<typename T, typename S, std::size_t M, std::size_t K, std::size_t N>
__global__ void mat_mul(T(&c)[M][N], S(&a)[M][K], S(&b)[K][N])
{
    for(int x = blockDim.x * blockIdx.x + threadIdx.x;
        x < M; x += gridDim.x * blockDim.x)
    {
        for(int y = blockDim.y * blockIdx.y + threadIdx.y;
            y < N; y += gridDim.y * blockDim.y)
        {
            // c[x][y] = a[x][y] + b[x][y];
            
            // c[x][y] = 0;

            // for(int k = 0; k < K; ++k)
            //     c[x][y] += a[x][k] * b[k][y];

            T sum = 0;

            for(int k = 0; k < K; ++k)
                sum += a[x][k] * b[k][y];

            c[x][y] = sum;
        }
    }
}

void test_cuda_kernel()
{
    constexpr std::size_t M = 5;
    constexpr std::size_t K = 4;
    constexpr std::size_t N = 5;

    constexpr std::size_t ThreadCount_x = 2;
    constexpr std::size_t ThreadCount_y = 2;

    constexpr std::size_t ByteSize = sizeof(int);
    
    /* A[5][4] x B[4][5] = C[5][5] */

    using matrix_a_t = int[M][K]; // A[5][4]
    using matrix_b_t = int[K][N]; // B[4][5]
    using matrix_c_t = int[M][N]; // C[5][5]

    int *a_buf, *b_buf, *c_buf;

    cudaMallocManaged(&a_buf, ByteSize * M * K); // A[M][K]
    cudaMallocManaged(&b_buf, ByteSize * K * N); // B[K][N]
    cudaMallocManaged(&c_buf, ByteSize * M * N); // C[M][N]

    auto generator = tcr::random_generator<int>(-5, 5);

    std::generate_n(a_buf,  M * K, std::ref(generator));
    std::generate_n(b_buf,  K * N, std::ref(generator));

    auto& A = (matrix_a_t&)a_buf[0];
    auto& B = (matrix_b_t&)b_buf[0];
    auto& C = (matrix_c_t&)c_buf[0];

    stream <<"A = " << A << endl;
    stream <<"B = " << B << endl;

    dim3 block_dim{ ThreadCount_x, ThreadCount_y }; // 블락 당 스레드 개수
    
    dim3 grid_dim // 블록 개수 >= 1
    {
        (unsigned int)(M + block_dim.x - 1) / block_dim.x,
        (unsigned int)(N + block_dim.y - 1) / block_dim.y,

        /*
            (5 + 2 -1)/ 2 = 3,
            (5 + 6 -1) / 6 = 1,
        */
    };

    // C = A x B
    mat_mul<<<grid_dim, block_dim>>>(C, A, B);
    cudaDeviceSynchronize();

    stream <<"C = " << C << endl;


    cudaFree(c_buf);
    cudaFree(b_buf);
    cudaFree(a_buf);

}

int main()
{
    test_cuda_kernel();
}
