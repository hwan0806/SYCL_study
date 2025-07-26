#include "cpg/cpg_std_extensions.hpp"
#include <sycl/sycl.hpp>

void test_default_device_info_basic()
{
    sycl::queue Q;

    auto device_default = Q.get_device();

    std::cout <<"Device name: " 
        << device_default.get_info<sycl::info::device::name>() << std::endl;

    std::cout <<"Is GPU: " 
        << device_default.is_gpu() << std::endl;
    
    std::cout <<"Is CPU: " 
        << device_default.is_cpu() << std::endl;

    std::cout <<"Is Host: " 
        << device_default.is_host() << std::endl;

    std::cout <<"Is Accelerator: " 
        << device_default.is_accelerator() << std::endl;
}

void test_cpu_device_info_basic()
{
    sycl::queue Q { sycl::cpu_selector{} };

    auto device = Q.get_device();

    std::cout <<"Device name: " 
        << device.get_info<sycl::info::device::name>() << std::endl;

    std::cout <<"Is GPU: " 
        << device.is_gpu() << std::endl;
    
    std::cout <<"Is CPU: " 
        << device.is_cpu() << std::endl;

    std::cout <<"Is Host: " 
        << device.is_host() << std::endl;

    std::cout <<"Is Accelerator: " 
        << device.is_accelerator() << std::endl;
}

/*
    
cl::sycl::info::device 
	https://intel.github.io/llvm-docs/doxygen/namespacecl_1_1sycl_1_1info.html#ab329ccdc28ac4187f67d14db9cbb6190

*/

void enumerate_device(sycl::platform const& platform)
{
    std::vector devices{ platform.get_devices() };

    for(sycl::device const& device: devices)
    {
        std::cout <<"\tDevice Name: "
            << device.get_info<sycl::info::device::name>() << std::endl;

        std::cout << "\tDevice Vendor: "
            << device.get_info<sycl::info::device::vendor>() << std::endl;

        std::cout << "\tMax Compute Units: "
            << device.get_info<sycl::info::device::max_compute_units>() << std::endl;

        auto max_group_size = device.get_info<sycl::info::device::max_work_group_size>();

        std::cout <<"\tMax Group Size : " << max_group_size << std::endl;
        
        sycl::id<3> max_work_item_sizes = 
            device.get_info<sycl::info::device::max_work_item_sizes>();

        std::cout <<"\tMax Item Count : {" 
                << max_work_item_sizes.get(0) <<", "
                << max_work_item_sizes.get(1) <<", " 
                << max_work_item_sizes.get(2) << " } " <<std::endl;

        auto sub_group_sizes =
            device.get_info<sycl::info::device::sub_group_sizes>();

        std::cout <<"\tMax Subgroup Sizes: " << sub_group_sizes << std::endl;

        auto local_mem_type = device.get_info<sycl::info::device::local_mem_type>();

        std::cout <<"\tLocal Mem Type: " << ( local_mem_type == cl::sycl::info::local_mem_type::local ?
            "local": ( local_mem_type == cl::sycl::info::local_mem_type::global ?
                "global" : "none")) << std::endl;

        std::cout <<"\tIs GPU: " << device.is_gpu() << std::endl;
        std::cout <<"\tIs CPU: " << device.is_cpu() << std::endl;
        std::cout <<"\tIs Accelerator: " << device.is_accelerator() << std::endl;

    }
}
void enumerate_platform()
{
    std::vector platforms { sycl::platform::get_platforms() };

    for(auto& platform: platforms)
    {
        std::cout << "Platform: "
            << platform.get_info<sycl::info::platform::name>() << std::endl;

        enumerate_device(platform);
        std::cout << std::endl;
    }
}

int main()
{
    std::cout << std::boolalpha;

    //test_default_device_info_basic();

    // test_cpu_device_info_basic();

    enumerate_platform();
}