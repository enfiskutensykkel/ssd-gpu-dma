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

set (no_sisci_samples false CACHE BOOL "Do not use SISCI for samples")
set (no_sisci_lib false CACHE BOOL "Do not use SISCI for library")

# API source files
file (GLOB api_source "${PROJECT_SOURCE_DIR}/src/*.c")
file (GLOB api_include "${PROJECT_BINARY_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.h")

include_directories ("${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${sisci_include}" "${sisci_include}/dis") 

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

# Macros for creating targets that depend on SISCI
macro (add_sisci_executable target binary deps files)
    if (sisci_include AND sisci_lib AND NOT no_sisci_samples)
        add_executable (${target} ${files})
        add_dependencies (${target} libnvme)
        target_compile_definitions (${target} PRIVATE _REENTRANT __DIS_CLUSTER__)
        target_link_libraries (${target} libnvme ${sisci_lib} ${deps})
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "${binary}")
        install (TARGETS ${target} DESTINATION "bin")
        list (APPEND sisci_targets ${target})
    endif ()
endmacro ()

# Build examples
add_subdirectory ("${PROJECT_SOURCE_DIR}/samples")

