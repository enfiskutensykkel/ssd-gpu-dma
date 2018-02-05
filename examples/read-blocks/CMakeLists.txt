cmake_minimum_required (VERSION 3.1)
project (libnvm-samples)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (read-blocks-module read-blocks "module.c;args.c;read.c")
set_multithread (read-blocks-module)

make_sample (read-blocks-smartio read-blocks "smartio.c;args.c;read.c")
set_multithread (read-blocks-smartio)
set_sisci (read-blocks-smartio)

make_samples_choice (read-blocks read-blocks-smartio read-blocks-module)
