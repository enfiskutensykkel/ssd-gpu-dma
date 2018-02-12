cmake_minimum_required (VERSION 3.1)
project (libnvm-samples)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (rpc-server rpc-server "rpc_server.c;util.c;segment.c")
set_sisci (rpc-server)

make_sample (rpc-identify rpc-identify "rpc_client.c;util.c;segment.c") 
set_sisci (rpc-identify)

add_custom_target(rpc DEPENDS rpc-server rpc-identify)
add_custom_target(rpc-client DEPENDS rpc-identify)
