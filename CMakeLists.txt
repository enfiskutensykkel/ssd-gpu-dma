cmake_minimum_required (VERSION 3.1)
project (libnvm LANGUAGES C CXX)

set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 11)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package (CUDA 8.0)


set (DIS "/opt/DIS" CACHE PATH "SISCI install location")
set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

set (nvidia_archs "30;50;52;60;61;70" CACHE STRING "NVIDIA compute architectures")
set (no_smartio_samples false CACHE BOOL "Do not use SISCI for examples")
set (no_smartio_benchmarks false CACHE BOOL "Do not use SISCI for benchmarks")
set (no_smartio false CACHE BOOL "Do not use SISCI for library or examples")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (no_cc false CACHE BOOL "Do not assume cache-coherent architecture")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")

if (no_smartio)
    set (no_smartio_samples true)
    set (no_smartio_benchmarks true)
endif ()

# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D__CUDA__;-std=c++11;${GPU_ARCHS}")
set (CUDA_NVCC_FLAGS_RELEASE "${CMAKE_NVCC_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()


# Try to find Nvidia driver
if (CUDA_FOUND AND NOT no_cuda)

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
find_path (sisci_include NAMES sisci_types.h sisci_api.h sisci_error.h PATHS "${DIS}/include" "${DIS}")
find_library (sisci_lib NAMES sisci PATHS "${DIS}/lib64" "${DIS}/lib" NO_DEFAULT_PATH)
if (sisci_lib)
    message(STATUS "Using DIS library ${sisci_lib}")
endif ()

# Library source files
set (libnvm_root "${PROJECT_SOURCE_DIR}/src")
file (GLOB libnvm_source "${libnvm_root}/*.c")
file (GLOB libnvm_dis_source "${libnvm_root}/dis/*.c")
file (GLOB libnvm_include "${PROJECT_BINARY_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.h")

# Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
file (GLOB module_source "${module_root}/*.c")

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")

# Samples shared files
set (samples_root "${CMAKE_SOURCE_DIR}/examples")



# Specify where stuff should be placed
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (module_output "${CMAKE_BINARY_DIR}/module")



# Set include directories
include_directories ("${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include")
if (sisci_include)
    include_directories ("${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${sisci_include}" "${sisci_include}/dis")
endif ()


# Create NVME library target
add_library (libnvm ${libnvm_source})
target_include_directories (libnvm PRIVATE "${libnvm_root}" "${libnvm_root}/dis")


if (sisci_include AND sisci_lib AND NOT no_smartio)
    target_sources (libnvm PRIVATE ${libnvm_dis_source})
    target_compile_definitions (libnvm PRIVATE _REENTRANT _SISCI __DIS_CLUSTER__)
    target_link_libraries (libnvm ${sisci_lib} Threads::Threads)

else ()
    target_link_libraries (libnvm Threads::Threads)

endif ()

if (CUDA_FOUND AND NOT no_cuda)
    target_compile_definitions(libnvm PRIVATE _CUDA __CUDA__)
endif ()

# Set name of library
#set_target_properties (libnvm PROPERTIES OUTPUT_NAME "${CMAKE_PROJECT_NAME}")
set_target_properties (libnvm PROPERTIES OUTPUT_NAME "nvm")

# Set install locations
install (FILES ${libnvm_include} DESTINATION "include")
install (TARGETS libnvm DESTINATION "lib")



# Create kernel module makefile
if (NOT no_module)
    if (CUDA_FOUND AND NOT no_cuda AND EXISTS "${driver_dir}/Module.symvers")
        set (module_ccflags "-D_CUDA -I${driver_include} -I${libnvm_root}")
        set (module_symbols "${driver_dir}/Module.symvers")
        message(STATUS "Configuring kernel module with CUDA")
    else ()
        set (module_ccflags "-I${libnvm_root}")
        message(STATUS "Configuring kernel module without CUDA")
    endif ()

    configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)

    add_custom_target (module "${CMAKE_MAKE_PROGRAM}" WORKING_DIRECTORY "${module_output}")
endif ()



# Macro for creating sample target
macro (make_sample target binary_name files)
    add_executable (${target} EXCLUDE_FROM_ALL ${files})

    add_dependencies (${target} libnvm)
    target_link_libraries (${target} libnvm)

    # TODO: Do we want to install samples or not?
    #set_target_properties (${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples")
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

    list (APPEND sample_targets "${target}")
    set (sample_targets "${sample_targets}" PARENT_SCOPE)

    #install (TARGETS ${target} DESTINATION "bin")
endmacro ()



# Macro for making a target depend on SISCI
macro (set_sisci target)
    if (NOT no_smartio_samples AND sisci_include AND sisci_lib)
        target_link_libraries (${target} ${sisci_lib})
        target_compile_definitions (${target} PRIVATE _REENTRANT __DIS_CLUSTER__)
    else ()
        list (REMOVE_ITEM sample_targets "${target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
        set_target_properties (${target} PROPERTIES EXCLUDE_FROM_ALL 1)
    endif ()
endmacro ()



# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} Threads::Threads)
endmacro ()


# Set preferred ordering for sample targets
macro (make_samples_choice ctarget sisci_target other_target)
    if (NOT no_smartio_samples AND sisci_include AND sisci_lib)
        add_custom_target (${ctarget} DEPENDS ${sisci_target})
        list (REMOVE_ITEM sample_targets "${other_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    else ()
        add_custom_target (${ctarget} DEPENDS ${other_target})
        list (REMOVE_ITEM sample_targets "${sisci_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    endif ()
endmacro ()



# Make benchmark target
# We assume all benchmarks are CUDA programs
macro (make_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda)
        cuda_add_executable (${target} EXCLUDE_FROM_ALL ${files} OPTIONS ${GPU_ARCHS} -D__CUDA__) # Ugly bugly
        target_compile_definitions (${target} PRIVATE __CUDA__)

        add_dependencies (${target} libnvm)
        target_link_libraries (${target} libnvm)
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
    endif ()
endmacro ()

macro (make_sisci_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda AND sisci_include AND sisci_lib AND NOT (no_smartio OR no_smartio_benchmarks))
        cuda_add_executable (${target} ${files} EXCLUDE_FROM_ALL OPTIONS ${GPU_ARCHS} -D__CUDA__ -D__DIS_CLUSTER__ -D_REENTRANT)
        target_compile_definitions(${target} PRIVATE __DIS_CLUSTER__ __CUDA__ _REENTRANT)

        add_dependencies (${target} libnvm)
        target_link_libraries (${target} libnvm ${sisci_lib})
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
    endif ()
endmacro ()

#macro (make_sisci_ptx prefix files)
#    cuda_compile_ptx (${prefix}_files "${files}" OPTIONS -D__CUDA__ -D__DIS_CLUSTER__ -D_REENTRANT)
#    add_custom_target (${prefix}-ptx DEPENDS ${prefix_files} SOURCES ${files})
#endmacro ()


# Set preferred ordering for benchmark targets
macro (make_benchmark_choice ctarget sisci_target other_target)
    if (NOT no_smartio_benchmarks AND sisci_include AND sisci_lib AND NOT no_smartio)
        add_custom_target (${ctarget} DEPENDS ${sisci_target})
        list (REMOVE_ITEM benchmark_targets "${other_target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)

    else ()
        add_custom_target (${ctarget} DEPENDS ${other_target})
        list (REMOVE_ITEM benchmark_targets "${sisci_target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)

    endif ()
endmacro ()



# Add individual samples
add_subdirectory ("${samples_root}/rpc")
add_subdirectory ("${samples_root}/read-blocks")
add_subdirectory ("${samples_root}/identify")
add_subdirectory ("${samples_root}/integrity")

# Build all samples
add_custom_target (samples DEPENDS ${sample_targets})
add_custom_target (examples DEPENDS samples)


# Add individual benchmarks
if (CUDA_FOUND)
    add_subdirectory ("${benchmarks_root}/latency")
    add_subdirectory ("${benchmarks_root}/cuda")
    add_custom_target (benchmarks DEPENDS ${benchmark_targets})
endif ()

