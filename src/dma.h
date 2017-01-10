#ifndef __SSD_DMA_DMA_H__
#define __SSD_DMA_DMA_H__

//#include <linux/module.h>
//#include <linux/fs.h>
//#include <asm/uaccess.h>
#include <linux/types.h>


struct remote_memory
{
    __u64                   virtual_addr;
};


struct dma_task
{
    struct remote_memory    memory_desc;    // remote memory descriptor
    size_t                  block_size;     // block size
};



#endif
