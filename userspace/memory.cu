#include <cuda.h>
#include <cunvme_ioctl.h>
#include "memory.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>


#define GPU_PAGE_SIZE   (1L << 16) // GPU bound pointers needs to be aligned to 64 KB


static int pin_memory(int fd, buffer_t* handle, int reqno)
{
    struct cunvme_pin* request;
   
    request = (struct cunvme_pin*) malloc(sizeof(struct cunvme_pin) + sizeof(uint64_t) * handle->n_addrs);
    if (request == NULL)
    {
        fprintf(stderr, "Failed to allocate ioctl request: %s\n", strerror(errno));
        return errno;
    }

    request->handle = CUNVME_NO_HANDLE;
    request->virt_addr = (unsigned long long) handle->virt_addr;
    request->num_pages = handle->n_addrs;

    int err = ioctl(fd, reqno, request);
    if (err < 0)
    {
        free(request);
        fprintf(stderr, "ioctl to kernel failed: %s\n", strerror(-err));
        return errno;
    }

    handle->kernel_handle = request->handle;
    handle->n_addrs = request->num_pages;
    for (long i = 0; i < request->num_pages; ++i)
    {
        handle->bus_addr[i] = request->bus_addr[i];
    }

    free(request);
    return 0;
}


static int unpin_memory(int fd, long kernel_handle)
{
    if (kernel_handle == CUNVME_NO_HANDLE)
    {
        return EINVAL;
    }

    struct cunvme_unpin request;
    request.handle = kernel_handle;

    int err = ioctl(fd, CUNVME_UNPIN, &request);
    if (err < 0)
    {
        fprintf(stderr, "ioctl to kernel failed: %s\n", strerror(-err));
        return errno;
    }

    return 0;
}


static long get_page_size(int dev)
{
    long page_size = GPU_PAGE_SIZE;

    if (dev < 0)
    {
        page_size = sysconf(_SC_PAGESIZE);

        if (page_size == -1)
        {
            fprintf(stderr, "Failed to get system page size: %s\n", strerror(errno));
        }
    }

    return page_size;
}


static int get_cuda_buffer(buffer_t* handle)
{
    cudaError_t err;

    err = cudaSetDevice(handle->device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        return EINVAL;
    }

    err = cudaMalloc(&handle->virt_addr, handle->range_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return ENOMEM;
    }

    unsigned int flag = 1;
    CUresult result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr) handle->virt_addr);
    if (result != CUDA_SUCCESS)
    {
        fprintf(stderr, "Failed to set memory synchronization option\n");
    }

    return 0;
}


extern "C"
buffer_t* get_buffer(int fd, int dev, size_t size)
{
    int err;

    long page_size = get_page_size(dev);
    if (page_size == -1)
    {
        return NULL;
    }

    size_t range_size = (size + page_size - 1) & ~(page_size - 1);
    size_t pages = range_size / page_size;

    buffer_t* handle = (buffer_t*) malloc(sizeof(buffer_t) + sizeof(uint64_t) * pages);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate memory handle: %s\n", strerror(errno));
        return NULL;
    }

    handle->kernel_handle = CUNVME_NO_HANDLE;
    handle->device = dev;
    handle->virt_addr = NULL;
    handle->range_size = range_size;
    handle->page_size = page_size;
    handle->n_addrs = pages;

    if (handle->device >= 0)
    {
        err = get_cuda_buffer(handle);
    }
    else
    {
        err = posix_memalign(&handle->virt_addr, handle->page_size, handle->range_size);
    }

    if (err != 0)
    {
        free(handle);
        fprintf(stderr, "Failed to allocate page-aligned memory buffer: %s\n", strerror(err));
        return NULL;
    }

    err = pin_memory(fd, handle, handle->device >= 0 ? CUNVME_PIN_GPU : CUNVME_PIN_RAM);
    if (err != 0)
    {
        if (handle->device >= 0)
        {
            cudaFree(handle->virt_addr);
        }
        else
        {
            free(handle->virt_addr);
        }
        free(handle);

        fprintf(stderr, "Failed to page-lock buffer pages: %s\n", strerror(err));
        return NULL;
    }

    return handle;
}


extern "C"
int get_page(int fd, int dev, page_t* page)
{
    long page_size = get_page_size(dev);
    if (page_size == -1)
    {
        return EIO;
    }

    buffer_t* handle = get_buffer(fd, dev, page_size);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate and pin page\n");
        return ENOMEM;
    }

    page->kernel_handle = handle->kernel_handle;
    page->device = handle->device;
    page->virt_addr = handle->virt_addr;
    page->page_size = handle->page_size;
    page->bus_addr = handle->bus_addr[0];

    free(handle);
    return 0;
}


extern "C"
void put_buffer(int fd, buffer_t* handle)
{
    if (handle != NULL)
    {
        unpin_memory(fd, handle->kernel_handle);
        
        if (handle->device >= 0)
        {
            cudaFree(handle->virt_addr);
        }
        else
        {
            free(handle->virt_addr);
        }

        free(handle);
    }
}


extern "C"
void put_page(int fd, page_t* page)
{
    if (page != NULL)
    {
        unpin_memory(fd, page->kernel_handle);
        page->kernel_handle = CUNVME_NO_HANDLE;

        if (page->device >= 0)
        {
            cudaFree(page->virt_addr);
        }
        else
        {
            free(page->virt_addr);
        }
        page->virt_addr = NULL;
    }
}

