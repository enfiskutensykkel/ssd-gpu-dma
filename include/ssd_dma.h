#ifndef __SSD_DMA_IOCTL_H__
#define __SSD_DMA_IOCTL_H__

#ifndef __KERNEL__
#define __user
#include <sys/types.h>
#endif

#include <linux/types.h>
#include <linux/ioctl.h>

#define MAGIC       0x80


/* Request types */
enum {
    SSD_DMA_START_TRANSFER  = _IO('S', MAGIC | 0x01),
    SSD_DMA_TRANSFER_STATUS = _IO('S', MAGIC | 0x02)
};


/* DMA vector element */
typedef struct {
    loff_t          file_offset;        /* offset in file */
    size_t          memory_offset;      /* offset in remote segment */
    size_t          length;             /* length of chunk */
} dma_vector_t;


/* Start DMA transfer request */
typedef struct {
    int             file_desc;          /* file descriptor of source file */
    __u64           io_address;         /* bus address of mapped memory */
    size_t          vector_length;      /* number of DMA vector elements */
    dma_vector_t    vector_elems[1];    /* DMA vector */
} dma_start_request_t;

#endif
