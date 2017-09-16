cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (rpc_server rpc_server "rpc_server.c")
set_sisci (rpc_server)

make_sample (rpc_client rpc_client "rpc_client.c") 
set_sisci (rpc_client)

