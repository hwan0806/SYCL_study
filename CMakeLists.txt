cmake_minimum_required(VERSION 3.8)

set(CMAKE_C_COMPILER "/opt/intel/oneapi/2025.1/bin/icx")
set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/2025.1/bin/icpx")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options(-fsycl)

set(CMAKE_CXX_STANDARD 20)
project(sycl_study LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(IntelSYCL REQUIRED)

add_subdirectory(practice)
add_subdirectory(example)

# add_executable(show_all_devices show_all_devices.cpp)
# add_sycl_to_target(
#   TARGET
#     show_all_devices
#   SOURCES
#     show_all_devices.cpp
# )

# add_executable(computation_test computation_test.cpp)
# add_sycl_to_target(
#   TARGET
#     computation_test
#   SOURCES
#     computation_test.cpp
# )

# add_executable(computation_test2 computation_test2.cpp)
# add_sycl_to_target(
#   TARGET
#     computation_test2
#   SOURCES
#     computation_test2.cpp
# )

#! only if you have installed nvidia cuda with codeplay plugin
# set(SYCL_DEVICE_FLAGS "-fsycl-targets=spir64,nvptx64-nvidia-cuda")
# target_compile_options(computation_test PRIVATE
#   ${SYCL_DEVICE_FLAGS}
# )
# target_link_options(computation_test PRIVATE
#   ${SYCL_DEVICE_FLAGS}
# )