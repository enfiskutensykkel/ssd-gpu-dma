#include <cuda.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "buffer.h"


/*
 * We need to use cudaHostAlloc, since posix_memalign does not work well with CUDA.
 */
MemoryPtr allocateHostMemory(size_t size)
{
    void* pointer = nullptr;

    auto err = cudaHostAlloc(&pointer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate memory in local RAM: ") + cudaGetErrorString(err));
    }

    memset(pointer, 0, size);
    return MemoryPtr(pointer, cudaFreeHost);
}

