#include <cuda.h>
#include "page.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <cunvme_ioctl.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>


static int unpin_memory(page_t* page, int fd)
{
    if (fd < 0)
    {
        fprintf(stderr, "Invalid ioctl fd: %d\n", fd);
        return EBADF;
    }
    else if (page->kernel_handle == CUNVME_NO_HANDLE)
    {
        fprintf(stderr, "Invalid kernel handle\n");
        return EBADF;
    }

    struct cunvme_unpin request;
    request.handle = page->kernel_handle;

    int err = ioctl(fd, CUNVME_UNPIN, &request);
    if (err < 0)
    {
        fprintf(stderr, "ioctl to kernel failed: %s\n", strerror(-err));
        return errno;
    }

    page->phys_addr = (unsigned long long) NULL;
    page->kernel_handle = -1;
    return 0;
}


static int pin_memory(page_t* page, int fd)
{
    if (fd < 0)
    {
        fprintf(stderr, "Invalid ioctl fd: %d\n", fd);
        return EBADF;
    }

    page->kernel_handle = CUNVME_NO_HANDLE;

    struct cunvme_pin request;
    request.handle = CUNVME_NO_HANDLE;
    request.paddr = (unsigned long long) NULL;
    request.vaddr = (unsigned long long) page->virt_addr;

    int err = ioctl(fd, CUNVME_PIN, &request);
    if (err < 0)
    {
        fprintf(stderr, "ioctl to kernel failed: %s\n", strerror(-err));
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

    page->device = -1;
    page->kernel_handle = CUNVME_NO_HANDLE;
    page->virt_addr = NULL;
    page->phys_addr = (uint64_t) NULL;
    page->page_size = 0;

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

    void* virt_addr;
    
    err = posix_memalign(&virt_addr, page_size, page_size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate page: %s\n", strerror(err));
        return err;
    }

    page->virt_addr = virt_addr;
    page->page_size = page_size;

    if (pin_memory(page, fd) != 0)
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
    else
    {
        unpin_memory(page, fd);
        free(page->virt_addr);
    }
}



#ifdef __cplusplus
}
#endif
