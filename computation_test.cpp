// bench_sycl.cpp  (C++17 / SYCL 2020)
// ------------------------------------------------------------
// 1) SGEMM (C = A × B)  --  GFLOPS
// 2) memcpy-like bandwidth                 --  GB/s
// ------------------------------------------------------------
#include <sycl/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace sycl;
using clk = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------------
// 커널 1:  SGEMM  (row-major, naive tiled implementation)
// -----------------------------------------------------------------------------
template<size_t TS>
class sgemm_kernel;

double run_sgemm(queue &q, size_t N)
{
    const size_t bytes = N * N * sizeof(float);
    // USM allocation
    float *A = malloc_device<float>(N * N, q);
    float *B = malloc_device<float>(N * N, q);
    float *C = malloc_device<float>(N * N, q);

    // 초기화 (host USM 사용)
    float *Ah = malloc_host<float>(N * N, q);
    float *Bh = malloc_host<float>(N * N, q);
    for (int i = 0; i < N * N; ++i) { Ah[i] = 1.0f;  Bh[i] = 1.0f; }

    q.memcpy(A, Ah, bytes).wait();
    q.memcpy(B, Bh, bytes).wait();

    // 타이밍 시작
    auto t0 = clk::now();

    // 단순 블록 곱셈 (TS x TS 타일)
    constexpr size_t TS = 16;           // 타일 크기 (SM/EVU에 맞게 수정)
    range<2> gws{N, N};
    range<2> lws{TS, TS};

    q.submit([&](handler &h){
        h.parallel_for<sgemm_kernel<TS>>(nd_range<2>{gws, lws},
        [=](nd_item<2> it)
        {
            int row = it.get_global_id(0);
            int col = it.get_global_id(1);
            float acc = 0.f;
            for (int k = 0; k < N; ++k)
                acc += A[row*N + k] * B[k*N + col];
            C[row*N + col] = acc;
        });
    }).wait();

    auto t1 = clk::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // FLOPs = 2 · N³
    double gflops = (2.0 * N * N * N) / (sec * 1e9);

    free(A, q); free(B, q); free(C, q);
    free(Ah, q); free(Bh, q);
    return gflops;
}

// -----------------------------------------------------------------------------
// 커널 2: 단순 memcpy (bandwidth 측정)
// -----------------------------------------------------------------------------
double run_memcpy(queue &q, size_t Nbytes)
{
    uint8_t *src = malloc_device<uint8_t>(Nbytes, q);
    uint8_t *dst = malloc_device<uint8_t>(Nbytes, q);

    // 타이밍
    auto t0 = clk::now();
    q.memcpy(dst, src, Nbytes).wait();
    auto t1 = clk::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double gbps = (double)Nbytes / (sec * 1e9);

    free(src, q); free(dst, q);
    return gbps;
}

void benchmark_on_device(const sycl::device& dev)
{
    try {
        sycl::queue q{dev};

        std::cout << "\nRunning on: "
                  << dev.get_info<sycl::info::device::name>() << '\n'
                  << "  Platform: " << dev.get_platform().get_info<sycl::info::platform::name>() << "\n"
                  << "  Backend : ";
                  switch (dev.get_backend()) {
                      case sycl::backend::opencl: std::cout << "OpenCL" << "\n"; break;
                      case sycl::backend::ext_oneapi_level_zero: std::cout << "Level Zero" << "\n"; break;
                      case sycl::backend::ext_oneapi_cuda: std::cout << "CUDA" << "\n"; break;
                      case sycl::backend::ext_oneapi_hip: std::cout << "HIP" << "\n"; break;
                      case sycl::backend::ext_oneapi_native_cpu: std::cout << "Native CPU" << "\n"; break;
                      default: std::cout << "Other" << "\n"; break;
                      sycl::default_selector_v(dev);
                    }


        // constexpr size_t N = 1024;
        constexpr size_t N = 1024 * 2;
        constexpr size_t BYTES = 256 * 1024 * 1024;

        // warm-up 1회
        run_sgemm(q, N);
        run_memcpy(q, BYTES);

        double gflops_sum = 0.0;
        double gbps_sum = 0.0;

        constexpr int TEST_NUM = 10;

        for (int i = 0; i < TEST_NUM; i++) {
            gflops_sum += run_sgemm(q, N);
            gbps_sum   += run_memcpy(q, BYTES);
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  [SGEMM]  N = " << N
                  << "  →  " << gflops_sum / TEST_NUM << " GFLOPS\n";
        std::cout << "  [Memcpy] " << BYTES / 1e6
                  << " MB  →  " << gbps_sum / TEST_NUM << " GB/s\n";
    }
    catch (const sycl::exception& e) {
        std::cerr << "  * Skipped (SYCL error: " << e.what() << ")\n";
    }
}

// -----------------------------------------------------------------------------
// 메인
// -----------------------------------------------------------------------------
int main()
{
  auto devices = sycl::device::get_devices();
  for(auto device : devices){
    benchmark_on_device(device);
  }
  return 0;
}