/*
    1. Basic - Basic Data-Parallel Kernels
    2. ND-range - Explicit ND-Range Parallel Kernels
    3. Hierarchical - Hierarchical Parallel Kernels
*/

#include "cpg/cpg_std_extensions.hpp"
#include "cpg/cpg_chrono_random.hpp"
#include <sycl/sycl.hpp>

namespace ccr = cpg::chrono_random;

void test_hierarchical_kernel()
{
    constexpr std::size_t M = 5;
    constexpr std::size_t K = 4;
    constexpr std::size_t N = 5;

    constexpr std::size_t blockDim_x = 2;
    constexpr std::size_t blockDim_y = 2;

    constexpr std::size_t gridDim_x 
        = (M + blockDim_x - 1) / blockDim_x;

    constexpr std::size_t gridDim_y 
        = (N + blockDim_y - 1) / blockDim_y;

    /* A[M][K] x B[K][N] = C[M][N] */
    using matrix_a_t = int[M][K]; // A[5][4]
    using matrix_b_t = int[K][N]; // B[4][5]
    using matrix_c_t = int[M][N]; // C[5][5]

    sycl::queue Q{ sycl::cpu_selector{} };

    int C[M][N];

    {
        sycl::buffer<int, 2> a_buf{ {M, K} };  // A[M][K]
        sycl::buffer<int, 2> b_buf{ {K, N} };  // B[K][N]
        sycl::buffer<int, 2> c_buf{ &C[0][0], {M, N} };

        auto generator = ccr::random_generator<int>(-5, 5);

        auto random = [&generator]()
        {
            return generator();
        };

        sycl::host_accessor a_h{a_buf}, b_h{b_buf};

        std::generate_n( &a_h[0][0], M * K, random);
        std::generate_n( &b_h[0][0], K * N, random);

        auto& A = (matrix_a_t&)a_h[0][0];
        auto& B = (matrix_b_t&)b_h[0][0];

        std::cout <<"A = " << A << std::endl;
        std::cout <<"B = " << B << std::endl;

        auto task = [&](sycl::handler& h)
        {
            sycl::accessor a{a_buf, h, sycl::read_only};
            sycl::accessor b{b_buf, h, sycl::read_only};
            sycl::accessor c{c_buf, h, sycl::write_only, sycl::no_init};

            // SYCL group size = CUDA의 블록 당 스레드 개수
            sycl::range group_size{blockDim_x, blockDim_y};

            // SYCL group count = CUDA의 블록 개수입니다
            sycl::range num_groups{ gridDim_x, gridDim_y };

            auto work_group = [=](sycl::group<2> group)
            {
                int blockIdx_x = group.get_id(0);
                int blockIdx_y = group.get_id(1);

                auto work_item = [&](sycl::h_item<2> hit)
                {
                    int threadIdx_x = hit.get_local_id(0);
                    int threadIdx_y = hit.get_local_id(1);

                    for(int x = blockDim_x * blockIdx_x + threadIdx_x;
                        x < M; x += gridDim_x * blockDim_x)
                    {
                        for(int y = blockDim_y * blockIdx_y + threadIdx_y;
                            y < N; y += gridDim_y * blockDim_y)
                        {
                            // c[x][y] = a[x][y] + b[x][y];

                            // c[x][y] = 0;

                            // for(int k = 0; k < K; ++k)
                            //     c[x][y] += a[x][k] * b[k][y];

                             int sum = 0;

                            for(int k = 0; k < K; ++k)
                                sum += a[x][k] * b[k][y];

                            c[x][y] = sum;
                        }
                    }

                };

                group.parallel_for_work_item(work_item);

            };

            h.parallel_for_work_group(num_groups, group_size, work_group);
        };

        Q.submit(task);
    }

    std::cout <<"C = " << C << std::endl;
}

int main()
{
    test_hierarchical_kernel();
}
