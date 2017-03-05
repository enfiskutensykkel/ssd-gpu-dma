#ifndef __CUNVME_REQUEST_H__
#define __CUNVME_REQUEST_H__

#ifndef __KERNEL__
#define __user
#endif

#include <asm/ioctl.h>


#define CUNVME_MAGIC            0x80
#define CUNVME_NO_HANDLE        -1L
#define CUNVME_NO_CUDA_DEVICE   -1


/* Supported operations */
enum
{
    CUNVME_PIN          = _IO('S', CUNVME_MAGIC | 0x01),
    CUNVME_UNPIN        = _IO('S', CUNVME_MAGIC | 0x02)
};


/* Pin request
 *
 * Find memory page, pin it in RAM and get the physical address.
 */
struct cunvme_pin
{
    long                handle;     /* out: kernel handle */
    unsigned long long  paddr;      /* out: physical address */
    unsigned long long  vaddr;      /* in:  virtual address */
};


/* Unpin request
 *
 * Release a previously pinned page.
 */
struct cunvme_unpin
{
    long                handle;     /* in: kernel handle */
};

#endif
