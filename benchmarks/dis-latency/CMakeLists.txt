cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)

make_sisci_benchmark (dis-latency-manager dis-latency-manager "manager.cu;segment.cu")
