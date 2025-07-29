#include "cpg/cpg_std_extensions.hpp"
#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

void test_cpg_std() {
  std::vector A{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  constexpr std::size_t N = 3;
  constexpr std::size_t M = 4;

  using matrix_t = int[N][M];

  // auto& A_3_4 = ( int(&)[N][M] )*A.data();
  auto &A_3_4 = (matrix_t &)*A.data();

  std::cout << "A     = " << A << std::endl;
  std::cout << "A_3_4 = " << A_3_4 << std::endl << std::endl;

  A_3_4[2][3] *= 2;

  std::cout << "A     = " << A << std::endl;
  std::cout << "A_3_4 = " << A_3_4 << std::endl;
}

void naive_matrix_multiplication() {
  constexpr std::size_t N = 3;
  constexpr std::size_t M = 3;
  constexpr std::size_t K = 3;

  int A[N][M]{{8, 2, 3}, {1, 7, 3}, {4, 2, 3}};
  int B[N][M]{{1, 4, 3}, {4, 2, 3}, {1, 5, 3}};
  int C[N][M]{};

  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "C = " << C << std::endl << std::endl;

  sycl::queue Q{sycl::default_selector{}};
  // sycl::queue Q{sycl::cpu_selector{}};

  sycl::range<2> num_items{N, M};

  // cpu memory(A, B, C)를 가지고 device memory(a_buf, b_buf, c_buf)를 생성
  sycl::buffer<int, 2> a_buf{&A[0][0], num_items};
  sycl::buffer<int, 2> b_buf{&B[0][0], num_items};
  sycl::buffer<int, 2> c_buf{&C[0][0], num_items};

  auto task = [&](sycl::handler &h) {
    // sycl::accessor a{a_buf, h};
    auto a = sycl::accessor{a_buf, h};
    auto b = sycl::accessor{b_buf, h};
    auto c = sycl::accessor{c_buf, h};

    // auto kernel = [=](sycl::id<2> idx)
    // item에는 세가지 종류 : 1. item 2. nd_item 3. h_item
    auto kernel = [=](sycl::item<2> idx) {
      // int n = idx[0];
      // int m = idx[1];

      int n = idx.get_id(0);
      int m = idx.get_id(1);

      int sum = 0;

      for (int k = 0; k < K; ++k)
        sum += a[n][k] * b[k][m];

      c[n][m] = sum;
    };

    h.parallel_for(num_items, kernel);
  };

  Q.submit(task);
  //! Q.wait()는 커널 완료만 보장. 데이터 전송은 별도 동기화 과정 필요
  //! cpu_selector 시엔 결과물 출력된 반면, default_selector 시엔 결과물 출력 안된 이유
  Q.wait();

  // SYCL buffer 동기화 규칙
  // ---------------------------------------------
  // 1. Host 메모리와 Device 메모리는 자동으로 동기화되지 않음
  // 2. Host에서 결과를 확인하려면 아래 중 하나 필요:
  //    (1) host_accessor 생성 시 자동 동기화 발생
  //    (2) buffer의 scope 종료 시 RAII에 의해 동기화 발생
  //    (3) queue.memcpy() 등 명시적 복사 수행 (USM의 경우)
  // ---------------------------------------------
  sycl::host_accessor h_c{c_buf};
  std::cout << "C = " << C << std::endl;
}

int main() {
  // test_cpg_std();

  naive_matrix_multiplication();
}
