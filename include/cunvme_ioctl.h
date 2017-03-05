#ifndef __CUNVME_REQUEST_H__
#define __CUNVME_REQUEST_H__


#ifndef __KERNEL__
#define __user
#include <stddef.h>
#include <stdint.h>
#endif

#include <asm/ioctl.h>


#define CUNVME_MAGIC            0x80
#define CUNVME_NO_HANDLE        -1L
#define CUNVME_NO_CUDA_DEVICE   -1


/* Supported operations */
enum
{
    CUNVME_VIRT_TO_PHYS     = _IO('S', CUNVME_MAGIC | 0x01),
};


/* Pin request
 *
 * Find memory page and get the physical address.
 */
struct cunvme_virt_to_phys
{
    uint64_t        paddr;          /* out: physical address */
    uint64_t        vaddr;          /* in:  virtual address */
};

#endif
