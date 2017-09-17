cmake_minimum_required (VERSION 3.1)
project (disnvme)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

# Set global compiler flags
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -DDEBUG" CACHE INTERNAL "" FORCE)

# Try to find SISCI
set (DIS "/opt/DIS" CACHE PATH "SISCI install location")
find_path (sisci_include NAMES sisci_types.h sisci_api.h sisci_error.h PATHS "${DIS}")
find_library (sisci_lib NAMES sisci PATHS "${DIS}/lib64" "${DIS}/lib")

set (no_sisci_examples false CACHE BOOL "Do not use SISCI for examples")
set (no_sisci_lib false CACHE BOOL "Do not use SISCI for library")

# API source files
file (GLOB api_source "${PROJECT_SOURCE_DIR}/src/*.c")
file (GLOB api_include "${PROJECT_BINARY_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.h")

include_directories ("${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${sisci_include}" "${sisci_include}/dis") 

# Samples shared files
set (samples_root "${CMAKE_SOURCE_DIR}/examples")
set (samples_common "${samples_root}/common")
file (GLOB samples_common_all "${samples_common}/*.c")
file (GLOB samples_common_sisci "${samples_common}/dis/*.c")


# Create NVME library target
add_library (libnvme SHARED ${api_source})

if (sisci_include AND sisci_lib AND NOT no_sisci_lib)
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

# Set install locations
install (FILES ${api_include} DESTINATION "include")
install (TARGETS libnvme DESTINATION "lib")

#macro (make_benchmark target binary_name files)
#    if (sisci_include AND sisci_lib)
#        add_executable (${target} EXCLUDE_FROM_ALL ${files})
#        list (APPEND benchmark_targets ${target})
#    endif ()
#endmacro ()


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


# Build samples
add_subdirectory ("${samples_root}/dis-rpc")
add_subdirectory ("${samples_root}/dis-rw")
add_subdirectory ("${samples_root}/identify")

# Build all samples
add_custom_target (examples DEPENDS ${sample_targets})
add_custom_target (samples DEPENDS examples)

#add_custom_target (benchmarks DEPENDS ${benchmark_targets})
