#include <algorithm>
#include <sycl/sycl.hpp>


// This function returns a vector of two (not necessarily distinct) devices,
// allowing computation to be split across said devices.
std::vector<sycl::device> get_devices() {
  std::vector<sycl::device> cpus;
  std::vector<sycl::device> gpus;
  
  // 모든 디바이스 가져오기
  auto devices = sycl::device::get_devices();
  
  // CPU와 GPU 분류
  for (const auto& dev : devices) {
    if (dev.is_cpu()) {
      cpus.push_back(dev);
    }
    else if (dev.is_gpu()) {
      gpus.push_back(dev);
    }
  }

  // 결과 출력
  std::cout << "발견된 CPU 개수: " << cpus.size() << std::endl;
  for (const auto& cpu : cpus) {
    std::cout << "CPU: " << cpu.get_info<sycl::info::device::name>() 
              << " (백엔드: " << cpu.get_info<sycl::info::device::driver_version>() << ")" << std::endl;
  }

  std::cout << "발견된 GPU 개수: " << gpus.size() << std::endl; 
  for (const auto& gpu : gpus) {
    std::cout << "GPU: " << gpu.get_info<sycl::info::device::name>() 
              << " (백엔드: " << gpu.get_info<sycl::info::device::driver_version>() << ")" << std::endl;
  }

  return devices;
}

std::vector<sycl::device> get_two_devices() {
  auto devs = sycl::device::get_devices();
  if (devs.size() == 0) throw "No devices available";
  if (devs.size() == 1) return {devs[0], devs[0]};
  return {devs[0], devs[1]};
}

int main() {
  constexpr size_t dataSize = 1024;
  constexpr float ratio = 0.5f;
  constexpr size_t dataSizeFirst = ratio * dataSize;
  constexpr size_t dataSizeSecond = dataSize - dataSizeFirst;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  auto devices = sycl::device::get_devices();
  for (const auto& dev : devices) {
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
      std::cout << "  Platform: " << dev.get_platform().get_info<sycl::info::platform::name>() << "\n";
      std::cout << "  Backend: ";
      switch (dev.get_backend()) {
        case sycl::backend::opencl: std::cout << "OpenCL"; break;
        case sycl::backend::ext_oneapi_level_zero: std::cout << "Level Zero"; break;
        case sycl::backend::ext_oneapi_cuda: std::cout << "CUDA"; break;
        case sycl::backend::ext_oneapi_hip: std::cout << "HIP"; break;
        case sycl::backend::ext_oneapi_native_cpu: std::cout << "Native CPU"; break;
        default: std::cout << "Other"; break;
        sycl::default_selector_v(dev);
      }

      auto size = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();

      std::cout << "\n";
      std::cout << "  Max Work Group Size: " << dev.get_info<sycl::info::device::max_work_group_size>() << "\n";
      std::cout << "  Max Compute Units: " << dev.get_info<sycl::info::device::max_compute_units>() << "\n";
      std::cout << "  Max Work Item Diemsions: " << dev.get_info<sycl::info::device::max_work_item_dimensions>() << "\n";
      std::cout << "  Max Work Item Sizes: " << size[0] << "x " << size[1] << "x " << size[2] << "\n";
      std::cout << "\n\n";
  }

  std::cout << "Running on devices:" << std::endl;
  auto devs = get_devices();
}