#include <cuda.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <cunvme_ioctl.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>


int lock_mem(int fd, void* vaddr, size_t size)
{
    struct cunvme_request request;
    request.handle = -1;
    request.device = CUNVME_NO_DEVICE;
    request.paddr = (uint64_t) NULL;
    request.vaddr = (uint64_t) vaddr;
    request.size = size;

    int err = ioctl(fd, CUNVME_PIN, &request);
    if (err < 0)
    {
        return err;
    }

    return request.handle;
}


int lock_dev_mem(int fd, int dev, void* addr, size_t size)
{
    cudaPointerAttributes attrs;

    cudaError_t cerr = cudaPointerGetAttributes(&attrs, addr);
    if (cerr != cudaSuccess)
    {
        return -EFAULT;
    }

    struct cunvme_request request;
    request.handle = -1;
    request.device = dev;
    request.paddr = (uint64_t) attrs.devicePointer;
    request.vaddr = (uint64_t) NULL;
    request.size = size;

    int kerr = ioctl(fd, CUNVME_PIN, &request);
    if (kerr < 0)
    {
        return kerr;
    }

    return request.handle;
}


uint64_t get_phys_addr(int fd, int handle)
{
    struct cunvme_request request;
    request.handle = handle;
    request.device = CUNVME_NO_DEVICE;
    request.paddr = (uint64_t) NULL;
    request.vaddr = (uint64_t) NULL;
    request.size = 0;

    int err = ioctl(fd, CUNVME_QUERY, &request);
    if (err < 0)
    {
        return (uint64_t) NULL;
    }

    return request.paddr;
}


int unlock_mem(int fd, int handle)
{
    struct cunvme_request request;
    request.handle = handle;
    request.device = CUNVME_NO_DEVICE;
    request.paddr = (uint64_t) NULL;
    request.vaddr = (uint64_t) NULL;
    request.size = 0;

    int err = ioctl(fd, CUNVME_UNPIN, &request);
    if (err < 0)
    {
        return err;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
