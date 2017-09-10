cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

add_sisci_executable (dis_manager nvm_dis_manager "Threads::Threads" "${PROJECT_SOURCE_DIR}/dis-manager/dis_manager.c")
#add_sisci_executable (util nvm_util "" "${PROJECT_SOURCE_DIR}/util/util.c")

#add_executable (native "${PROJECT_SOURCE_DIR}/native/test.c")
#target_link_libraries (native libnvme)
#install (TARGETS native DESTINATION "bin")

add_custom_target (sisci-examples DEPENDS ${sisci_targets})

