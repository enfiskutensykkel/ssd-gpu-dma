#ifndef __SSD_DMA_NVME_H__
#define __SSD_DMA_NVME_H__

#include <linux/fs.h>
#include <linux/scatterlist.h>
#include "bind.h"


enum nvme_func {
    NVME_FREE_IOD = 0,
    NVME_NUM_FUNCS
};


/* Array containing NVMe driver functions */
extern func_t nvme_funcs[];

/* Forward declaration of NVMe device handle */
typedef struct nvme_ns* dev_handle_t;


/* Get a NVMe device handle from a file pointer */
dev_handle_t get_nvme_device_handle(struct file* file);


/* Release a NVMe device handle */
void put_nvme_device_handle(dev_handle_t handle);

#endif
