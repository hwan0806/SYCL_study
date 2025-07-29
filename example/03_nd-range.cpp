#include "cpg/cpg_std_extensions.hpp"
#include "cpg/cpg_chrono_random.hpp"
#include <sycl/sycl.hpp>

//! CUDA Rule : [gridDim][blockDim] - [블락개수][스레드개수] - [blockIdx][threadIdx]
// CUDA Rule #1 – [gridDim][blockDim] = [블럭개수][스레드개수] = [blockIdx][threadIdx]
// CUDA Rule #2 – CUDA 커널 루프다! SYCL 커널 루프다!
//
// 1. Basic Data-Parallel Kernels - Basic
//    item, range, parallel_for
//
// 2. Explicit ND-Range Parallel Kernels – ND-Range
//! nd_item이 핵심!
//    nd_item, nd_range, range, group, (work-group) – CUDA에서는 block
//    parallel_for work group
//
// 3. Hierarchical Parallel Kernels – Hierarchical
//    h_item, nd_range, range
//    parallel_for_work_group
//    parallel_for_work_item
//
// CUDA와 SYCL 개념 대응
// ----------------------------------------------------------
// CUDA: warp  -> 쓰레드 묶음 = 32 threads
// SYCL: sub-group -> 16 threads
//
// CUDA: block -> SYCL: work-group
// CUDA: threadIdx -> SYCL: local_index
//
// CUDA: global index 계산 예시
//    x = blockDim.x * blockIdx.x + threadIdx.x;
//    y = blockDim.y * blockIdx.y + threadIdx.y;
//
// SYCL: global index 계산 예시
//    x = get_global_id(0);
//    y = get_global_id(1);
//
// CUDA: local index (block)
// int m = threadIdx.x;
// int n = threadIdx.y;
//
// SYCL: local index (work-group)
// int m = nd_item.get_local_id(0);
// int n = nd_item.get_local_id(1);
//
// CUDA: block
// __shared__           -> 공유 메모리
// __syncthreads()      -> 스레드 블록 내 동기화
//
// SYCL: work-group
// sycl::local_accessor -> work-group 내 스레드 간 공유하는 메모리 (로컬 메모리)
// nd_item.barrier()    -> work-group 내 스레드 동기화
//
// SYCL: sycl::queue가 있고,
// event를 이용해서 task graph 작성 가능
//
// CUDA: stream -> sycl::queue와 대응
// ==============================================
// int A[3 * 4]; // 전체 element 개수 = 12
// int B[3 * 4];
// int C[3 * 4];
//
// void kernel(C, A, B)
// {
//     C[i] = A[i] + B[i];
// }
//
// CUDA: kernel<<<3, 4>>>(C, A, B);
//       3 - 블록 개수, 4 - 블록당 스레드 개수
//       [gridDim][blockDim] = 3 * 4 = 12
//
// SYCL: h.parallel_for(range{ {12}, {4} }, kernel);
//
// ==============================================
// ==============================================
// int A[3 * 4 + 2]; // 전체 element 개수 = 14 : Execution range : 각각의 element는 하나의 element를 이룸
// int B[3 * 4 + 2];
// int C[3 * 4 + 2];
//
// void cuda_kernel(C, A, B)
// {
//     for(int i = blockDim * blockIdx + threadIdx;
//             i < (3*4+2);
//             i += gridDim * blockDim)
//     {
//         C[i] = A[i] + B[i];
//     }
// }
//
// CUDA: kernel<<<3, 4>>>(C, A, B);
//       3 - 블록 개수, 4 - 블록당 스레드 개수
//       [gridDim][blockDim] = 3 * 4 = 12
//
//! sycl kernel이 하나의 item을 구성하고 (아래의 i), 이 item이 총 14개가 되는 것
// void sycl_kernel(C, A, B)
// {
//     int i = nd_item.get_global_id(0);
//     C[i] = A[i] + B[i];
// }
//
// SYCL: h.parallel_for(
//           nd_range{ {14}, {4} }, // 글로벌 14개, 로컬 4개
//           sycl_kernel
//       );
//
// 아래와 같이 표현 가능
// range<1> global_range{14};
// range<1> local_range{4};  // work-group 당 스레드 개수 - CUDA의 blockDim과 같은 개념
// SYCL: h.parallel_for(
//           nd_range{ global_range, local_range }, // 글로벌 14개, 로컬 4개
//           sycl_kernel
//       );

// ==============================================

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
        v.data(); // v에 멤버함수 data() 존재하는가?를 확인하고 처리
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

        // buffer는 cpu 메모리와 gpu 메모리 사이에 서로 공유할 수 있는 메모리 영역
        sycl::buffer<int, 2> a_buf{a.data(), global_range };
        sycl::buffer<int, 2> b_buf{b.data(), global_range };
        sycl::buffer<int, 2> c_buf{c.data(), global_range };

        auto task = [&](sycl::handler& h)
        {
            sycl::accessor a_acc{a_buf, h, sycl::read_only};
            sycl::accessor b_acc{b_buf, h, sycl::read_only};
            sycl::accessor c_acc{c_buf, h, sycl::write_only};

            sycl::stream output{ 1024 * 8, 1024,  h};

            // 이 커널은 다수의 스레드에 의해 비동기적으로 병렬 실행
            //! 아래 부분이 하이라이트
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

        // queue의 submit 작업 시, event를 리턴. 이는 나중에 task graph 관리에 사용
        auto event = Q.submit(task);
    }
    //! {} 블럭으로 감싼 이유
    // block을 벗어날 때, Q보다 buffer의 destructor가 먼저 호출됨. 
    // 이 때, Q의 task가 실행중이라면 buffer의 destructor가 기다리다가, 데이터가 릴리즈되면(task에서 c_acc가 device로부터 해방되면 그 메모리를 cpu 메모리에다가 복사해줌. )
    // 따라서 block을 사용하면 submit 이후 wait을 안해줘도 됨.

    std::cout <<"C = " << C << std::endl;
}

int main()
{
    // test_ccr_random_generator();

    // test_cast_to_reference();

    test_nd_range_kernel();
}

