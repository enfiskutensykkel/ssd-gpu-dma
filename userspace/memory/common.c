#include <cunvme_ioctl.h>
#include "types.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>


int pin_memory(int fd, memory_t* handle, int reqno)
{
    struct cunvme_pin* request;
   
    request = malloc(sizeof(struct cunvme_pin) + sizeof(uint64_t) * handle->n_addrs);
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


int unpin_memory(int fd, long kernel_handle)
{
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

