// #include <cpg/cpg_std_extensions.hpp>
#include <sycl/sycl.hpp>


void test_hello_sycl()
{
	std::cout << "Hello, SYCL" << std::endl;
}

void test_sycl()
{
	sycl::queue queue{ sycl::gpu_selector{} };

	auto device = queue.get_device();
	auto name = device.get_info<sycl::info::device::name>();
	std::cout << "Device Name: " << name << std::endl;

	sycl::range<1> num_items{ 10 }; // items ������ 10����

	sycl::buffer<int> gpu_memory{ num_items }; // int[10]

	auto work = [&](sycl::handler& handler)
	{
		auto acc = sycl::accessor{ gpu_memory,  handler};

		sycl::stream out{ 1024 * 32, 1024, handler};

		auto kernel = [=](auto i) // item, nd_item, h_item
		{
			out << "i = " << i.get_id() << "\n";

			acc[i] = i + 1; // device - gpu, cpu, FPGA, AI accelerator device
		};


		handler.parallel_for(num_items, kernel);
	};

	queue.submit(work);

	// host_accessor's constructor�� blocking operation
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
