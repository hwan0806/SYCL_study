#include "cpg/cpg_std_extensions.hpp"
#include "cpg/cpg_chrono_random.hpp"
#include <sycl/sycl.hpp>

// 굉장히 많이 사용할 것입니다.
namespace ccr = cpg::chrono_random;

void test_ccr_random_generator()
{
    std::vector<float> V(100);

    std::cout <<"V = " << V << std::endl;

    auto generator = ccr::random_generator<int>(-5, 5);

    // ccr::random_fill(V, generator);
    ccr::random_parallel_fill(V, generator);

    std::cout <<"V = " << V << std::endl;
}

template<typename T> // T = matrix_t
T& cast_reference(auto& v)
    requires requires
    {
        v.data(); // v에 멤버함수 data() 존재하는가?
    }
{
    // auto& V_3_4 = (matrix_t&)*V.data();
    return (T&)*v.data();
}

void test_cast_to_reference()
{
    constexpr std::size_t M = 3;
    constexpr std::size_t N = 4;

    // float[M][N]
    std::vector<float> V( M * N );

    auto generator = ccr::random_generator<int>(-10, 10);

    ccr::random_fill(V, generator);
   
    using matrix_t = float[M][N];
    using array_t = float[N][M];

    // auto& V_3_4 = (matrix_t&)*V.data();

    auto& V_3_4 = cast_reference<matrix_t>(V);
    auto& A_4_3 = cast_reference<array_t>(V);

    std::cout <<"V     = " << V << std::endl;
    std::cout <<"V_3_4 = " << V_3_4 << std::endl;
    std::cout <<"A_4_3 = " << A_4_3 << std::endl;
}

/*
    nd_item, nd_range, global_range, local_range, work_group
*/

void test_nd_range_kernel()
{
    // G는 전체 element 개수이고요, G는 global range라는 뜻에서 G로 나타낸 것입니다
    constexpr std::size_t G = 5;

    // L은 local을 나타내는 것이구요, work-group 당 item 개수이고요
    // CUDA의 block 당 스레드 개수 - blockDim과 같은 개념입니다
    // item -> 하나의 단일 스레드에 의해 실행됩니다.
    // item -> 하나의 element를 연산합니다 
    constexpr std::size_t L = 2;

    // G x G = 5 x 5 = int[5][5]
    std::vector<int> a(G * G), b(G * G), c(G * G);

    using matrix_t = int[G][G];

    auto generator = ccr::random_generator<int>(-5, 5);
    
    ccr::random_fill(a, generator);
    ccr::random_fill(b, generator);

    auto& A = cast_reference<matrix_t>(a);
    auto& B = cast_reference<matrix_t>(b);
    auto& C = cast_reference<matrix_t>(c);

    std::cout <<"A = " << A << std::endl;
    std::cout <<"B = " << B << std::endl;
    std::cout <<"C = " << C << std::endl;

    {
        sycl::queue Q{ sycl::cpu_selector{} };

        sycl::range<2> global_range{G, G}; // {5, 5}
        sycl::range<2> local_range{L, L};

        sycl::buffer<int, 2> a_buf{a.data(), global_range };
        sycl::buffer<int, 2> b_buf{b.data(), global_range };
        sycl::buffer<int, 2> c_buf{c.data(), global_range };

        auto task = [&](sycl::handler& h)
        {
            sycl::accessor a_acc{a_buf, h, sycl::read_only};
            sycl::accessor b_acc{b_buf, h, sycl::read_only};
            sycl::accessor c_acc{c_buf, h, sycl::write_only};

            sycl::stream output{ 1024 * 8, 1024,  h};

            auto kernel = [=](sycl::nd_item<2> nit)
            {
                //  x = blockDim.x * blockIdx.x + threadIdx.x
                int x = nit.get_global_id(0);

                //  y = blockDim.y * blockIdx.y + threadIdx.y
                int y = nit.get_global_id(1);

                // m = threadIdx.x
                int m = nit.get_local_id(0);

                // n = threadIdx.y
                int n = nit.get_local_id(1);

                output << "{ global: " << x <<", " << y<<" } - "
                    << "{ work-groupd: " << m <<", " << n <<" }\n";

                // c_acc[x][y] = a_acc[x][y] + b_acc[x][y];

                int sum = 0;

                for(int k = 0; k < G; ++k)
                    sum += a_acc[x][k] * b_acc[k][y];

                c_acc[x][y] = sum;
            };

            h.parallel_for(sycl::nd_range{global_range, local_range}, kernel);
            // h.parallel_for(sycl::nd_range<2>{ {G, G}, {L, L}}, kernel);

        };

        auto event = Q.submit(task);
    }

    std::cout <<"C = " << C << std::endl;
}

int main()
{
    // test_ccr_random_generator();

    // test_cast_to_reference();

    test_nd_range_kernel();
}

