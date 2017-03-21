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
    CUNVME_PIN_RAM      = _IO('S', CUNVME_MAGIC | 0x01),
    CUNVME_PIN_GPU      = _IO('S', CUNVME_MAGIC | 0x02),
    CUNVME_UNPIN        = _IO('S', CUNVME_MAGIC | 0x03),
};


/* Pin memory region
 *
 * Find pages associated to a memory range, page-lock them and return
 * valid DMA addresses.
 */
struct cunvme_pin
{
    long                handle;         /* out: kernel handle */
    unsigned long long  virt_addr;      /* in:  virtual address of memory region*/
    long                num_pages;      /* inout: number of pages to pin */
    unsigned long long  bus_addr[0];    /* out: array of bus addresses */
};


/* Unpin memory region
 *
 * Release a previously pinned memory region.
 */
struct cunvme_unpin
{
    long                handle;         /* in: kernel handle */
};


#endif
