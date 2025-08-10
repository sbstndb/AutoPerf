cmake_minimum_required(VERSION 3.20)
project(AutoPerfCpp LANGUAGES CXX)

add_compile_options(-O3 -march=native -mtune=native -mavx2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories for libraries and executables
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

# ------------------------------------------------------------
# Options to control which kernels/targets are built
# ------------------------------------------------------------
option(AUTOPERF_BUILD_ALL "Build all known kernels and their targets" OFF)
set(AUTOPERF_KERNELS "" CACHE STRING "Semicolon-separated list of kernels to build (e.g., matmul;reduce)")
option(AUTOPERF_ENABLE_BENCHES "Build benchmark executables" ON)
option(AUTOPERF_ENABLE_TESTS "Build test executables" ON)
option(AUTOPERF_ENABLE_DEMOS "Build demo main executables" ON)
option(AUTOPERF_ENABLE_OPENMP "Enable OpenMP support (compile and link)" ON)

# List of known kernels in the repo
set(AUTOPERF_KNOWN_KERNELS matvec matmul reduce search custom)

# ------------------------------------------------------------
# Conditional dependencies
# ------------------------------------------------------------
if(AUTOPERF_ENABLE_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

if(AUTOPERF_ENABLE_BENCHES)
  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

# OpenMP (optional)
if(AUTOPERF_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling flags and linking for CXX")
    set(_AUTOPERF_OMP_TGT OpenMP::OpenMP_CXX)
  else()
    message(WARNING "AUTOPERF_ENABLE_OPENMP=ON but OpenMP CXX not found. Proceeding without OpenMP.")
    set(_AUTOPERF_OMP_TGT "")
  endif()
endif()

# ------------------------------------------------------------
# Kernel definition helper (OBJECT library + optional targets)
# ------------------------------------------------------------
function(autoperf_define_kernel name)
  set(src_file "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_${name}.cpp")
  if(NOT EXISTS "${src_file}")
    message(WARNING "Kernel source not found for ${name}: ${src_file}")
    return()
  endif()

  add_library(kernel_${name}_obj OBJECT ${src_file})
  target_include_directories(kernel_${name}_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(kernel_${name}_obj PRIVATE -O3 -march=native -mtune=native -mavx2)
  if(AUTOPERF_ENABLE_OPENMP AND _AUTOPERF_OMP_TGT)
    # Ensure the object library is compiled with OpenMP flags
    target_link_libraries(kernel_${name}_obj PUBLIC ${_AUTOPERF_OMP_TGT})
    target_compile_definitions(kernel_${name}_obj PUBLIC AUTOPERF_USE_OPENMP=1)
  endif()

  if(AUTOPERF_ENABLE_BENCHES)
    set(bench_src "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_${name}.cpp")
    if(EXISTS "${bench_src}")
      add_executable(bench_${name} ${bench_src})
      target_link_libraries(bench_${name} PRIVATE kernel_${name}_obj benchmark::benchmark benchmark::benchmark_main)
      target_compile_options(bench_${name} PRIVATE -O3 -march=native -mtune=native -mavx2)
      if(AUTOPERF_ENABLE_OPENMP AND _AUTOPERF_OMP_TGT)
        target_link_libraries(bench_${name} PRIVATE ${_AUTOPERF_OMP_TGT})
      endif()
    endif()
  endif()

  if(AUTOPERF_ENABLE_TESTS)
    set(test_src "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${name}.cpp")
    if(EXISTS "${test_src}")
      add_executable(tests_${name} ${test_src})
      target_link_libraries(tests_${name} PRIVATE kernel_${name}_obj GTest::gtest_main)
      target_compile_options(tests_${name} PRIVATE -O3 -march=native -mtune=native -mavx2)
      if(AUTOPERF_ENABLE_OPENMP AND _AUTOPERF_OMP_TGT)
        target_link_libraries(tests_${name} PRIVATE ${_AUTOPERF_OMP_TGT})
      endif()
      gtest_discover_tests(tests_${name})
    endif()
  endif()

  if(AUTOPERF_ENABLE_DEMOS)
    set(main_src "${CMAKE_CURRENT_SOURCE_DIR}/src/main_${name}.cpp")
    if(EXISTS "${main_src}")
      add_executable(main_${name} ${main_src})
      target_link_libraries(main_${name} PRIVATE kernel_${name}_obj)
      target_compile_options(main_${name} PRIVATE -O3 -march=native -mtune=native -mavx2)
      if(AUTOPERF_ENABLE_OPENMP AND _AUTOPERF_OMP_TGT)
        target_link_libraries(main_${name} PRIVATE ${_AUTOPERF_OMP_TGT})
      endif()
    endif()
  endif()
endfunction()

# ------------------------------------------------------------
# Select and define kernels
# ------------------------------------------------------------
if(AUTOPERF_BUILD_ALL)
  set(_AUTOPERF_SELECTED_KERNELS ${AUTOPERF_KNOWN_KERNELS})
elseif(AUTOPERF_KERNELS)
  set(_AUTOPERF_SELECTED_KERNELS ${AUTOPERF_KERNELS})
else()
  set(_AUTOPERF_SELECTED_KERNELS "")
  message(STATUS "No kernels selected. Set AUTOPERF_KERNELS or enable AUTOPERF_BUILD_ALL.")
endif()

foreach(k IN LISTS _AUTOPERF_SELECTED_KERNELS)
  autoperf_define_kernel(${k})
endforeach()
