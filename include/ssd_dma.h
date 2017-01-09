#ifndef __SSD_DMA_IOCTL_H__
#define __SSD_DMA_IOCTL_H__

#ifndef __KERNEL__
#define __user
#include <stddef.h>
#endif

#include <asm/ioctl.h>
#include <linux/ioctl.h>

#define MAGIC       0x80

enum {
    SSD_DMA_START_TRANSFER  = _IO('S', MAGIC | 0x01),
    SSD_DMA_TRANSFER_STATUS = _IO('S', MAGIC | 0x02)
};


/* Start a DMA transfer (SSD_DMA_START_TRANSFER) */
struct start_job
{
    int             file_desc;  // file descriptor to file on local SSD
    size_t          block_size; // block size
    size_t          num_blocks; // number of blocks to transfer
    size_t          file_pos;   // offset in file (in blocks)
    volatile void*  mem_ptr;    // handle to remote memory
    size_t          offset;     // offset into remote memory
};

#endif
