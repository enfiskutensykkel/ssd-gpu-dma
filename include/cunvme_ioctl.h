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
    CUNVME_PIN      = _IO(CUNVME_MAGIC | 'S', 0x01),
};


/* Pin request
 *
 * Find memory page, pin it in memory and get the physical address.
 */
struct cunvme_pin_page
{
    long            handle;         /* out: reference handle */
    uint64_t        paddr;          /* out: physical address */
    uint64_t        vaddr;          /* in:  virtual address */
};

#endif
