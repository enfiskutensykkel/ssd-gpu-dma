cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

make_sample (dis_rpc_server dis_rpc_manager "rpc_server.c")
set_sisci (dis_rpc_server)

make_sample (dis_rpc_client dis_rpc_client "rpc_client.c") 
set_sisci (dis_rpc_client)

