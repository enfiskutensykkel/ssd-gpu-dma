#include <cunvme_ioctl.h>
#include "ram.h"
#include "types.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>


extern int pin_memory(int fd, memory_t* handle, int reqno);

extern int unpin_memory(int fd, long handle);


memory_t* get_ram_buffer(int fd, size_t size)
{
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
        fprintf(stderr, "Failed to get system page size: %s\n", strerror(errno));
        return NULL;
    }

    size_t range_size = (size + page_size - 1) & ~(page_size - 1);
    size_t n_addrs = range_size / page_size;

    memory_t* handle = malloc(sizeof(memory_t) + sizeof(uint64_t) * n_addrs);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate memory handle: %s\n", strerror(errno));
        return NULL;
    }

    handle->kernel_handle = CUNVME_NO_HANDLE;
    handle->device = -1;
    handle->virt_addr = NULL;
    handle->range_size = range_size;
    handle->page_size = page_size;
    handle->n_addrs = n_addrs;

    int err = posix_memalign(&handle->virt_addr, handle->page_size, handle->range_size);
    if (err != 0)
    {
        free(handle);
        fprintf(stderr, "Failed to allocated page aligned memory: %s\n", strerror(err));
        return NULL;
    }

    err = pin_memory(fd, handle, CUNVME_PIN_RAM);
    if (err != 0)
    {
        free(handle->virt_addr);
        free(handle);
        fprintf(stderr, "Failed to page-lock memory: %s\n", strerror(err));
        return NULL;
    }

    return handle;
}


void put_ram_buffer(int fd, memory_t* handle)
{
    if (handle != NULL)
    {
        unpin_memory(fd, handle->kernel_handle);
        free(handle->virt_addr);
        free(handle);
    }
}


int get_ram_page(int fd, page_t* page)
{
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
        fprintf(stderr, "Failed to get system page size: %s\n", strerror(errno));
        return errno;
    }

    memory_t* handle = get_ram_buffer(fd, page_size);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate and pin page\n");
        return ENOMEM;
    }
    
    page->kernel_handle = handle->kernel_handle;
    page->device = -1;
    page->virt_addr = handle->virt_addr;
    page->page_size = handle->page_size;
    page->bus_addr = handle->bus_addr[0];

    free(handle);
    return 0;
}


void put_ram_page(int fd, page_t* page)
{
    if (page != NULL)
    {
        unpin_memory(fd, page->kernel_handle);
        free(page->virt_addr);
    }
}
