// #include <cpg/cpg_std_extensions.hpp>
#include <sycl/sycl.hpp>


void test_hello_sycl()
{
	std::cout << "Hello, SYCL" << std::endl;
}

void test_sycl()
{
	sycl::queue queue{ sycl::default_selector{} };

	auto device = queue.get_device();
	auto name = device.get_info<sycl::info::device::name>();
	std::cout << "Device Name: " << name << std::endl;

	sycl::range<1> num_items{ 10 }; // items 개수가 10개
	sycl::buffer<int> gpu_memory{ num_items }; // int[10] 만큼 메모리 할당(device)

	auto work = [&](sycl::handler& handler)
	{
		auto acc = sycl::accessor{ gpu_memory,  handler};	// device 메모리 접근
		sycl::stream out{ 1024 * 32, 1024, handler};

		// device에서 작동하는 커널 함수
		// SYCL 장치 코드는 host 메모리를 참조할 수 없으므로 lvalue-ref([&])가 아닌 [=] 캡처 사용
		// i는 현재 sycl::item<1> 타입
		// sycl::item<1>은 sycl에서 커널 내부 실행중인 work-item을 식별하는 객체.
		// <N>은 N차원 인덱스 공간에서 실행됨을 의미
		// sycl::item은 다음 정보 제공
		// 1. 현재 work-item의 인덱스(get_id)
		// 2. 현재 work-item의 범위(get_range)
		auto kernel = [=](auto item) 
		{
			// std::cout 사용 불가 => sycl::stream으로 버퍼 만들고 이를 디바이스 통해 출력
			out << "i = " << item.get_id() << "\n";

			acc[item] = item + 1; // device - gpu, cpu, FPGA, AI accelerator device
		};


		handler.parallel_for(num_items, kernel);
	};

	// submit을 통해 커널 함수 비동기 실행
	queue.submit(work);

	// SYCL 동기화 방식 비교
	// ------------------------------------------------------------
	// | 방식                   | 동기화 범위     | 동작 시점               |
	// | --------------------- | ---------- | ------------------- |
	// | queue.wait()          | 큐 전체       | 호출 즉시               |
	// | sycl::host_accessor   | 해당 buffer만 | host accessor 생성 시점 |
	// ------------------------------------------------------------
	// 일반적으로 host_accessor를 쓰면 따로 queue.wait()를 안 해도 안전.
	// 여러 buffer를 다루거나, 명시적 동기화가 필요할 때는 queue.wait() 사용 필요.

	//! queue.wait();

	// host_accessor 생성 시 장치 연산이 완료될 때까지 자동으로 동기화(blocking: host 스레드가 대기)
	sycl::host_accessor h{ gpu_memory };

	for (int i = 0; i < 10; ++i)
		std::cout << h[i] << ", ";

	std::cout << std::endl;
}

int main()
{
	// test_hello_sycl();

	test_sycl();
}
