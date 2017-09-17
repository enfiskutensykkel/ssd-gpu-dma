cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (no_sisci_examples false CACHE BOOL "Do not use SISCI for examples")
set (no_sisci false CACHE BOOL "Do not use SISCI for library")
set (no_module false CACHE BOOL "Do not build kernel module")

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)
find_package (CUDA 8.0)

# Set global compiler flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -DDEBUG" CACHE INTERNAL "" FORCE)

# Try to find Nvidia headers (will choose the newest version)
file (GLOB_RECURSE nvheader_paths "/usr/src/nvidia-*/nv-p2p2.h")

if (CUDA_FOUND AND NOT nvheader_paths)
    message (WARNING "Could not find Nvidia header files")
else ()
    list (REVERSE nvheader_paths)
    list (GET nvheader_paths 0 nv_header)
endif ()

set (nv_header "${nv_header}" CACHE STRING "Path to Nvidia p2p header")

# Attempt to find Nvidia driver
get_filename_component (nvdriver_path "${nv_header}" DIRECTORY)
if (NOT EXISTS "${nvdriver_path}/Module.symvers")
    get_filename_component (nvdriver_path "${nvdriver_path}" DIRECTORY)
endif ()

set (nv_driver "${nvdriver_path}/Module.symvers")

if (CUDA_FOUND AND NOT EXISTS "${nv_driver}")
    message (WARNING "Could not find Nvidia driver symbols")
    set (nv_driver "")
endif ()

set (nv_driver "${nv_driver}" CACHE STRING "Path to Nvidia driver symbols")


# Try to find SISCI
set (DIS "/opt/DIS" CACHE PATH "SISCI install location")
find_path (sisci_include NAMES sisci_types.h sisci_api.h sisci_error.h PATHS "${DIS}")
find_library (sisci_lib NAMES sisci PATHS "${DIS}/lib64" "${DIS}/lib")

# API source files
file (GLOB libnvme_source "${PROJECT_SOURCE_DIR}/src/*.c")
file (GLOB libnvme_include "${PROJECT_BINARY_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.h")

# Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
file (GLOB module_source "${module_root}/*.c")

# Samples shared files
set (samples_root "${CMAKE_SOURCE_DIR}/examples")
set (samples_common "${samples_root}/common")
file (GLOB samples_common_all "${samples_common}/*.c")
file (GLOB samples_common_sisci "${samples_common}/dis/*.c")


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

# Set name of library
set_target_properties (libnvme PROPERTIES OUTPUT_NAME "${CMAKE_PROJECT_NAME}")

# Specify where stuff should be placed
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (module_output "${CMAKE_BINARY_DIR}/module")

# Set install locations
install (FILES ${libnvme_include} DESTINATION "include")
install (TARGETS libnvme DESTINATION "lib")


# Create kernel module target
if (NOT no_module)
    #message(STATUS "${CUDA_TOOLKIT_ROOT_DIR}")
    #add_custom_command (OUTPUT nvdriver 
    set (kdir "/lib/modules/${CMAKE_SYSTEM_VERSION}/build")
    configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
endif ()


# Macro for creating sample binaries
macro (make_sample target binary_name files)
    add_executable (${target} EXCLUDE_FROM_ALL ${files})

    # Make sample depend on library
    add_dependencies (${target} libnvme)
    target_link_libraries (${target} libnvme)

    # Add shared example code
    target_include_directories (${target} PRIVATE "${samples_common}")
    target_sources (${target} PRIVATE ${samples_common_all})

    set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary_name}")

    list (APPEND sample_targets "${target}")
    set(sample_targets "${sample_targets}" PARENT_SCOPE)
endmacro ()


# Macro for making a sample depend on SISCI
macro (set_sisci target)
    if (NOT no_sisci_examples AND sisci_include AND sisci_lib)
        target_link_libraries (${target} ${sisci_lib})
        target_sources (${target} PRIVATE ${samples_common_sisci})
        target_compile_definitions (${target} PRIVATE _REENTRANT __DIS_CLUSTER__)
    else ()
        list (REMOVE_ITEM sample_targets "${target}")
        set(sample_targets "${sample_targets}" PARENT_SCOPE)
        set_target_properties (${target} PROPERTIES EXCLUDE_FROM_ALL 1)
    endif ()
endmacro ()


#macro (make_benchmark target binary_name files)
#    if (sisci_include AND sisci_lib)
#        add_executable (${target} EXCLUDE_FROM_ALL ${files})
#        list (APPEND benchmark_targets ${target})
#    endif ()
#endmacro ()



# Build samples
add_subdirectory ("${samples_root}/dis-rpc")
add_subdirectory ("${samples_root}/dis-rw")
add_subdirectory ("${samples_root}/identify")

# Build all samples
add_custom_target (examples DEPENDS ${sample_targets})
add_custom_target (samples DEPENDS examples)

#add_custom_target (benchmarks DEPENDS ${benchmark_targets})
