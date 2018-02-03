cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

make_sisci_benchmark (latency-benchmark-smartio latency-bench "main.cu;settings.cu;buffer.cu;ctrl.cc;queue.cc;barrier.cc;transfer.cc")

make_benchmark (latency-benchmark-module latency-bench "main.cu;settings.cu;buffer.cu;ctrl.cc;queue.cc;barrier.cc;transfer.cc")

make_benchmark_choice (latency-benchmark latency-benchmark-smartio latency-benchmark-module)
