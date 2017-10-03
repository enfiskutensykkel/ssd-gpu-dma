cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

make_benchmark (simple-rdma simple-rdma "main.cu;dma.cu;queue.cc;transfer.cc;benchmark.cu")
