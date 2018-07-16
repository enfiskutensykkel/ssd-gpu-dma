cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

# CUDA targets doesn't support target_include_directories
#include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include" )
include_directories ("include")

file (GLOB module_files "src/*.cu" "src/*.cc")
file (GLOB smartio_files "src/*.cu" "src/*.cc" "smartio/*.cu" "smartio/*.cc")

make_benchmark (latency-benchmark-module latency-bench "${module_files}")

make_sisci_benchmark (latency-benchmark-smartio latency-bench "${smartio_files}")

make_benchmark_choice (latency-benchmark latency-benchmark-smartio latency-benchmark-module)
