#include <sycl/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace sycl;
using clk = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------------
// GPU-friendly tiled SGEMM kernel using shared local memory
// -----------------------------------------------------------------------------
template<int TS>
class TiledSGEMM;

double run_sgemm(queue &q, size_t N)
{
    const size_t bytes = N * N * sizeof(float);

    float *A = malloc_device<float>(N * N, q);
    float *B = malloc_device<float>(N * N, q);
    float *C = malloc_device<float>(N * N, q);

    float *Ah = malloc_host<float>(N * N, q);
    float *Bh = malloc_host<float>(N * N, q);
    for (int i = 0; i < N * N; ++i) { Ah[i] = 1.0f;  Bh[i] = 1.0f; }

    q.memcpy(A, Ah, bytes).wait();
    q.memcpy(B, Bh, bytes).wait();

    constexpr int TS = 16;
    range<2> gws{N, N};
    range<2> lws{TS, TS};

    auto t0 = clk::now();

    q.submit([&](handler &h) {
        local_accessor<float, 2> Asub({TS, TS}, h);
        local_accessor<float, 2> Bsub({TS, TS}, h);

        h.parallel_for<TiledSGEMM<TS>>(
            nd_range<2>{gws, lws}, [=](nd_item<2> it) {
                int row = it.get_global_id(0);
                int col = it.get_global_id(1);
                int local_row = it.get_local_id(0);
                int local_col = it.get_local_id(1);

                float acc = 0.0f;
                for (int t = 0; t < N / TS; ++t) {
                    Asub[local_row][local_col] = A[row * N + t * TS + local_col];
                    Bsub[local_row][local_col] = B[(t * TS + local_row) * N + col];
                    it.barrier(access::fence_space::local_space);

                    for (int k = 0; k < TS; ++k) {
                        acc += Asub[local_row][k] * Bsub[k][local_col];
                    }
                    it.barrier(access::fence_space::local_space);
                }
                C[row * N + col] = acc;
            });
    }).wait();

    auto t1 = clk::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double gflops = (2.0 * N * N * N) / (sec * 1e9);

    free(A, q); free(B, q); free(C, q);
    free(Ah, q); free(Bh, q);
    return gflops;
}

// -----------------------------------------------------------------------------
double run_memcpy(queue &q, size_t Nbytes)
{
    uint8_t *src = malloc_device<uint8_t>(Nbytes, q);
    uint8_t *dst = malloc_device<uint8_t>(Nbytes, q);

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
    // if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) return; // GPU에 유리하게 필터링

    try {
        sycl::queue q{dev};

        std::cout << "\nRunning on: "
                  << dev.get_info<info::device::name>() << '\n'
                  << "  Platform: " << dev.get_platform().get_info<info::platform::name>() << "\n"
                  << "  Backend : Level Zero\n";

        constexpr size_t N = 2048;  // 큰 문제 크기
        constexpr size_t BYTES = 256 * 1024 * 1024;

        run_sgemm(q, N);  // warm-up
        run_memcpy(q, BYTES);

        double gflops_sum = 0.0;
        double gbps_sum = 0.0;
        constexpr int TEST_NUM = 10;

        for (int i = 0; i < TEST_NUM; ++i) {
            gflops_sum += run_sgemm(q, N);
            gbps_sum   += run_memcpy(q, BYTES);
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  [SGEMM]  N = " << N << "  →  " << gflops_sum / TEST_NUM << " GFLOPS\n";
        std::cout << "  [Memcpy] " << BYTES / 1e6 << " MB  →  " << gbps_sum / TEST_NUM << " GB/s\n";
    }
    catch (const sycl::exception& e) {
        std::cerr << "  * Skipped (SYCL error: " << e.what() << ")\n";
    }
}

int main()
{
    auto devices = sycl::device::get_devices();
    for (auto& device : devices) {
        benchmark_on_device(device);
    }
    return 0;
}
