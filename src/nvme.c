#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/nvme.h>
#include <linux/types.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/scatterlist.h>
#include <linux/kthread.h>
#include <linux/blk-mq.h>
#include <linux/blkdev.h>
#include <linux/kthread.h>
#include <asm/page.h>
#include <linux/kref.h>
#include <uapi/linux/nvme_ioctl.h>
#include <bind.h> /* include generated stub */
#include "nvme.h"



static inline struct nvme_ns* get_ns_from_disk(struct gendisk* disk)
{
    struct nvme_ns* ns;
    ns = (struct nvme_ns*) disk->private_data;
    return ns;
}


dev_handle_t get_nvme_device_handle(struct file* fp)
{
    struct inode* inode = fp->f_inode;
    struct super_block* sb = inode->i_sb;
    struct block_device* bdev = sb->s_bdev;
    struct gendisk* bd_disk = bdev->bd_disk;
    struct nvme_ns* ns;
    
    ns = get_ns_from_disk(bd_disk);
    if (ns == NULL)
    {
        printk(KERN_ERR "Could not find NVMe namespace\n");
        return NULL;
    }

    return ns;
}


void put_nvme_device_handle(dev_handle_t handle)
{
}


