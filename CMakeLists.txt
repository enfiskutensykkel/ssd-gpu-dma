cmake_minimum_required (VERSION 3.1)
project (disnvme LANGUAGES C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 11)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package (CUDA 8.0)


set (DIS "/opt/DIS" CACHE PATH "SISCI install location")
set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")

set (no_sisci_examples false CACHE BOOL "Do not use SISCI for examples")
set (no_sisci false CACHE BOOL "Do not use SISCI for library")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_nvidia_driver false CACHE BOOL "Build kernel module without NVIDIA support")
set (no_cuda false CACHE BOOL "Do not build with CUDA")


# Set global compiler flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -Wextra -pedantic -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D__CUDA__")
set (CUDA_NVCC_FLAGS_RELEASE "${CMAKE_NVCC_FLAGS_RELEASE} -D__CUDA__ -O2")



# Try to find Nvidia driver
if (CUDA_FOUND AND NOT no_nvidia_driver)

    # Try to find Nvidia headers (will choose the newest version)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-p2p.h")
    string (REPLACE "nv-p2p.h" "" driver_paths "${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()

    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")

    # Find Nvidia driver symbols
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()

    message(STATUS "Using NVIDIA driver found in ${driver_dir}")
endif ()



# Try to find SISCI
find_path (sisci_include NAMES sisci_types.h sisci_api.h sisci_error.h PATHS "${DIS}")
find_library (sisci_lib NAMES sisci PATHS "${DIS}/lib64" "${DIS}/lib" NO_DEFAULT_PATH)
if (sisci_lib)
    message(STATUS "Using DIS library ${sisci_lib}")
endif ()

# Library source files
set (libnvme_root "${PROJECT_SOURCE_DIR}/src")
file (GLOB libnvme_source "${libnvme_root}/*.c")
file (GLOB libnvme_include "${PROJECT_BINARY_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.h")

# Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
file (GLOB module_source "${module_root}/*.c")

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")

# Samples shared files
set (samples_root "${CMAKE_SOURCE_DIR}/examples")
set (samples_common "${samples_root}/common")
file (GLOB samples_common_all "${samples_common}/*.c")
file (GLOB samples_common_sisci "${samples_common}/dis/*.c")



# Specify where stuff should be placed
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (module_output "${CMAKE_BINARY_DIR}/module")



# Set include directories
include_directories ("${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${sisci_include}" "${sisci_include}/dis") 


# Create NVME library target
add_library (libnvme SHARED ${libnvme_source})

if (sisci_include AND sisci_lib AND NOT no_sisci)
    target_compile_definitions(libnvme PRIVATE _REENTRANT _SISCI __DIS_CLUSTER__)
    target_link_libraries (libnvme ${sisci_lib} Threads::Threads)
else ()
    target_link_libraries (libnvme Threads::Threads)
endif ()

if (CUDA_FOUND AND NOT no_cuda)
    target_compile_definitions(libnvme PRIVATE _CUDA __CUDA__)
endif ()

# Set name of library
set_target_properties (libnvme PROPERTIES OUTPUT_NAME "${CMAKE_PROJECT_NAME}")

# Set install locations
install (FILES ${libnvme_include} DESTINATION "include")
install (TARGETS libnvme DESTINATION "lib")





# Create kernel module makefile
if (NOT no_module)
    if (CUDA_FOUND AND NOT no_nvidia_driver AND EXISTS "${driver_dir}/Module.symvers")
        set (module_ccflags "-D_CUDA -I${driver_include} -I${libnvme_root}")
        set (module_symbols "${driver_dir}/Module.symvers")
    else ()
        set (module_ccflags "-I${libnvme_root}")
    endif ()

    set (kdir "/lib/modules/${CMAKE_SYSTEM_VERSION}/build")
    configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
endif ()



# Macro for creating sample target
macro (make_sample target binary_name files)
    add_executable (${target} EXCLUDE_FROM_ALL ${files})

    add_dependencies (${target} libnvme)
    target_link_libraries (${target} libnvme)

    target_include_directories (${target} PRIVATE "${samples_common}")
    target_sources (${target} PRIVATE ${samples_common_all})

    set_target_properties (${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples")
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary_name}")

    list (APPEND sample_targets "${target}")
    set (sample_targets "${sample_targets}" PARENT_SCOPE)
endmacro ()



# Macro for making a target depend on SISCI
macro (set_sisci target)
    if (NOT no_sisci_examples AND sisci_include AND sisci_lib)
        target_link_libraries (${target} ${sisci_lib})
        target_sources (${target} PRIVATE ${samples_common_sisci})
        target_compile_definitions (${target} PRIVATE _REENTRANT __DIS_CLUSTER__)
    else ()
        if (sample_targets)
            list (REMOVE_ITEM sample_targets "${target}")
            set (sample_targets "${sample_targets}" PARENT_SCOPE)
        endif ()

        if (benchmark_targets)
            list (REMOVE_ITEM benchmark_targets "${target}")
            set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
        endif ()
        set_target_properties (${target} PROPERTIES EXCLUDE_FROM_ALL 1)
    endif ()
endmacro ()



# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()



# Make benchmark target
# We assume all benchmarks are CUDA programs
macro (make_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda)
        cuda_add_executable (${target} ${files} OPTIONS -O3 -gencode arch=compute_50,code=sm_50 -D__CUDA__ -std=c++11) # Ugly bugly

        add_dependencies (${target} libnvme)
        target_link_libraries (${target} libnvme)

        set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary_name}")

        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)

        install (TARGETS ${target} DESTINATION "bin")
    endif ()
endmacro ()



# Add individual samples
add_subdirectory ("${samples_root}/dis-rpc")
add_subdirectory ("${samples_root}/dis-rw")
add_subdirectory ("${samples_root}/read-example")
add_subdirectory ("${samples_root}/identify")

# Build all samples
add_custom_target (samples DEPENDS ${sample_targets})
add_custom_target (examples DEPENDS samples)


# Add individual benchmarks
if (CUDA_FOUND)
    add_subdirectory ("${benchmarks_root}/simple-rdma")
    add_custom_target (benchmarks DEPENDS ${benchmark_targets})
endif ()

