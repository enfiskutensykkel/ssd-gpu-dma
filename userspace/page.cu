#include <cuda.h>
#include "page.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <cunvme_ioctl.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>



static int pin_ram_page(page_t* page, int fd)
{
    struct cunvme_pin_page request;
    request.handle = CUNVME_NO_HANDLE;
    request.paddr = (uint64_t) NULL;
    request.vaddr = (uint64_t) page->virt_addr;

    if (ioctl(fd, CUNVME_PIN, &request) < 0)
    {
        fprintf(stderr, "ioctl to kernel failed: %s\n", strerror(errno));
        return errno;
    }

    page->kernel_handle = request.handle;
    page->phys_addr = request.paddr;
    return 0;
}


static int get_gpu_page(page_t* page, size_t size, int fd, int dev)
{
    // TODO: Copy magic from rdma bench
    return 0;
}


int get_page(page_t* page, int fd, int dev)
{
    int err;

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
        fprintf(stderr, "Failed to retrieve page size: %s\n", strerror(errno));
        return errno;
    }
    
    if (dev >= 0)
    {
        return get_gpu_page(page, page_size, fd, dev);
    }

    void* addr = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == NULL)
    {
        fprintf(stderr, "Failed to mmap page: %s\n", strerror(errno));
        return errno;
    }

    if (mlock(addr, page_size) != 0)
    {
        fprintf(stderr, "Failed to mlock page: %s\n", strerror(errno));
        err = errno;
        munmap(addr, page_size);
        return err;
    }

    page->device = -1;
    page->kernel_handle = CUNVME_NO_HANDLE;
    page->virt_addr = addr;
    page->phys_addr = (uint64_t) NULL;
    page->page_size = page_size;

    if (pin_ram_page(page, dev) != 0)
    {
        put_page(page, -1);
        return EIO;
    }

    return 0;
}


static void put_gpu_page(page_t* page, int fd)
{
}


void put_page(page_t* page, int fd)
{
    if (page->device >= 0)
    {
        put_gpu_page(page, fd);
        return;
    }

    if (fd > 0)
    {
    }

    munlock(page->virt_addr, page->page_size);
    munmap(page->virt_addr, page->page_size);
}



#ifdef __cplusplus
}
#endif
