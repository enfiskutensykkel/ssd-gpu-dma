cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (dis_rw_example dis_rw "rw.c")
set_sisci (dis_rw_example)
