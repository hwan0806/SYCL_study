/*
    2022.03.02. 현재 CUDA C++20을 지원하지 않습니다.
*/

#include <tpf_output.hpp>
#include <tpf_chrono_random.hpp>
#include <cuda.h>

namespace tcr = tpf::chrono_random;

tpf::sstream stream;
auto& endl = tpf::endl;

template<typename T, std::size_t M>
__global__ void mat_mul( T(&c)[M][M], const T(&a)[M][M], const T(&b)[M][M])
{
    // 행의 인덱스 - global index : SYCL의 get_global_id(0)
    int xx = blockDim.x * blockIdx.x + threadIdx.x;

    // 열의 인덱스 - global index : SYCL의 get_global_id(1)
    int yy = blockDim.y * blockIdx.y + threadIdx.y;

    for(int x = xx; x < M; x += gridDim.x * blockDim.x)
    {
        for(int y = yy; y < M; y += gridDim.y * blockDim.y)
        {
            // c[x][y] = a[x][y] + b[x][y];

            // CUDA의 threadIdx : SYCL의 get_local_id
            printf("{ global: %d, %d } - { local: %d, %d } \n",
                x, y,threadIdx.x, threadIdx.y);

            T sum = 0;

            for(int k = 0; k < M; ++k)
                sum += a[x][k] * b[k][y];

            c[x][y] = sum;
        }
    }
}

void test_cuda_kernel()
{
    constexpr std::size_t G = 5;
    constexpr std::size_t ElementCount = G * G;

    constexpr std::size_t ByteCount = sizeof(int) * ElementCount;

    int *c, *a, *b;

    cudaMallocManaged(&c, ByteCount);
    cudaMallocManaged(&a, ByteCount);
    cudaMallocManaged(&b, ByteCount);

    auto generator = tcr::random_generator<int>(-5, 5);

    std::generate_n(a, ElementCount, std::ref(generator));
    std::generate_n(b, ElementCount, std::ref(generator));
    
    using matrix_t = int[G][G];

    auto& A = (matrix_t&)a[0];
    auto& B = (matrix_t&)b[0];
    auto& C = (matrix_t&)c[0];

    dim3 grid_dim{ 2, 2 }; // block의 개수
    dim3 block_dim{ 2, 2 }; // block 당 스레드 개수

    // gridDim.x * blockDim.x = 2 x 2 = 4 행
    // gridDim.y * blockDim.y = 2 x 2 = 4 열

    mat_mul<<<grid_dim, block_dim>>>(C, A, B);

    cudaDeviceSynchronize();    // CUDA kernel이 끝날때까지 대기

    stream << "A = " << A << endl;
    stream << "B = " << B << endl;
    stream << "C = " << C << endl;

    cudaFree(b);
    cudaFree(a);
    cudaFree(c);
}

int main()
{
    test_cuda_kernel();
}