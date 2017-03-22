#include <cuda.h>
#include <cunvme_ioctl.h>
#include "types.h"
#include "gpu.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <errno.h>

#define GPU_PAGE_SIZE   (1UL << 16) // Pointers need to be 64KB aligned


extern "C" int pin_memory(int fd, memory_t* handle, int reqno);

extern "C" int unpin_memory(int fd, long handle);


extern "C"
memory_t* get_gpu_buffer(int fd, int dev, size_t size)
{
    cudaError_t err;

    err = cudaSetDevice(dev);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    size_t range_size = (size + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    printf("size=%zu range_size=%zu\n", size, range_size);
    size_t n_addrs = range_size / GPU_PAGE_SIZE;
    size_t handle_size = sizeof(memory_t) + sizeof(uint64_t) * n_addrs;

    memory_t* handle = (memory_t*) malloc(handle_size);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate memory handle: %s\n", strerror(errno));
        return NULL;
    }

    handle->kernel_handle = CUNVME_NO_HANDLE;
    handle->device = dev;
    handle->virt_addr = NULL;
    handle->range_size = range_size;
    handle->page_size = GPU_PAGE_SIZE;
    handle->n_addrs = n_addrs;

    err = cudaMalloc(&handle->virt_addr, handle->range_size);
    if (err != cudaSuccess)
    {
        free(handle);
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, handle->virt_addr);
    if (err != cudaSuccess)
    {
        cudaFree(handle->virt_addr);
        free(handle);
        fprintf(stderr, "Failed to get pointer attributes: %s\n", cudaGetErrorString(err));
        return NULL;
    }

    unsigned int flag = 1;
    CUresult s = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) attrs.devicePointer);
    if (s != CUDA_SUCCESS)
    {
        fprintf(stderr, "Failed to set memory synchronization\n");
    }

    int status = pin_memory(fd, handle, CUNVME_PIN_GPU);
    if (status != 0)
    {
        cudaFree(handle->virt_addr);
        free(handle);
        fprintf(stderr, "Failed to pin device memory: %s\n", strerror(status));
        return NULL;
    }
        
    return handle;
}


extern "C"
void put_gpu_buffer(int fd, memory_t* handle)
{
    if (handle != NULL)
    {
        unpin_memory(fd, handle->kernel_handle);
        cudaFree(handle->virt_addr);
        free(handle);
    }
}


extern "C"
int get_gpu_page(int fd, int dev, page_t* page)
{
    memory_t* handle = get_gpu_buffer(fd, dev, GPU_PAGE_SIZE);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate and pin page\n");
        return ENOMEM;
    }

    page->kernel_handle = handle->kernel_handle;
    page->device = dev;
    page->virt_addr = handle->virt_addr;
    page->page_size = handle->page_size;
    page->bus_addr = handle->bus_addr[0];

    free(handle);
    return 0;
}


extern "C"
void put_gpu_page(int fd, page_t* page)
{
    if (page != NULL)
    {
        unpin_memory(fd, page->kernel_handle);
        cudaFree(page->virt_addr);
    }
}
