
# SYCL 실행 파일을 생성하는 함수 정의
function(add_sycl_executable TARGET_NAME)
  set(SOURCE_FILE "${TARGET_NAME}.cpp")
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  add_sycl_to_target(
    TARGET
      ${TARGET_NAME}
    SOURCES
      ${SOURCE_FILE}
  )
  
  # TBB 라이브러리 찾기 및 링킹
  find_package(TBB REQUIRED)
  target_link_libraries(${TARGET_NAME} TBB::tbb)
endfunction()

add_sycl_executable(01_HelloSycl)
add_sycl_executable(02_mat2d)
add_sycl_executable(03_nd-range)
add_sycl_executable(04_h-item)
# add_sycl_executable(05_device)


#! only if you have installed nvidia cuda with codeplay plugin
# set(SYCL_DEVICE_FLAGS "-fsycl-targets=spir64,nvptx64-nvidia-cuda")
# target_compile_options(computation_test PRIVATE
#   ${SYCL_DEVICE_FLAGS}
# )
# target_link_options(computation_test PRIVATE
#   ${SYCL_DEVICE_FLAGS}
# )