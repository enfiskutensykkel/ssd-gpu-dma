#ifndef __SSD_DMA_NVME_H__
#define __SSD_DMA_NVME_H__

#include <linux/types.h>
#include <linux/scatterlist.h>

struct nvme_iod {
    unsigned long private;
    int npages;
    int offset;
    int nents;
    int length;
    dma_addr_t first_dma;
    struct scatterlist meta_sg[1];
    struct scatterlist sg[0];
};


struct nvme_iod* create_iod(const struct start_transfer*);


#endif
