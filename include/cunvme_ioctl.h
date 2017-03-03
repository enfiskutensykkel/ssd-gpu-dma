#ifndef __CUNVME_REQUEST_H__
#define __CUNVME_REQUEST_H__


#ifndef __KERNEL__
#define __user
#include <stddef.h>
#include <stdint.h>
#endif

#include <asm/ioctl.h>


#define CUNVME_MAGIC            0x80
#define CUNVME_NO_DEVICE        -1


/* Supported operations */
enum
{
    CUNVME_QUERY                = _IO(CUNVME_MAGIC | 'S', 0x00),
    CUNVME_PIN                  = _IO(CUNVME_MAGIC | 'S', 0x01),
    CUNVME_UNPIN                = _IO(CUNVME_MAGIC | 'S', 0x02)
};


/* Multi-purpose request */
struct cunvme_request
{
    int         handle;         /* pinned memory handle */
    int         device;         /* indicates if memory is located on a CUDA device */
    uint64_t    paddr;          /* physical address */
    uint64_t    vaddr;          /* virtual address or device memory address */
    size_t      size;           /* size of memory chunck */
};

#endif
