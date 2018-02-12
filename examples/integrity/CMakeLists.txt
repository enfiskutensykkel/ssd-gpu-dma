cmake_minimum_required (VERSION 3.1)
project (libnvm-samples)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (integrity-smartio integrity-util "integrity.c;transfer.c;util.c")
set_multithread (integrity-smartio)
set_sisci (integrity-smartio)


make_sample (integrity-module integrity-util "integrity.c;transfer.c;util.c")
set_multithread (integrity-module)


make_samples_choice (integrity-util integrity-smartio integrity-module)
add_custom_target (integrity DEPENDS integrity-util)
