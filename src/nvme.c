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
#include "nvme.h"


/* Linux obama 4.4.21jonas #26 SMP Wed Dec 7 11:15:24 CET 2016 x86_64 x86_64 x86_64 GNU/Linux */
struct nvme_ctrl {
    const struct nvme_ctrl_ops *ops;
    struct request_queue *admin_q;
    struct device *dev;
    struct kref kref;
    int instance;
    struct blk_mq_tag_set *tagset;
    struct list_head namespaces;
    struct mutex namespaces_mutex;
    struct device *device;  /* char device */
    struct list_head node;

    char name[12];
    char serial[20];
    char model[40];
    char firmware_rev[8];

    u32 ctrl_config;

    u32 page_size;
    u32 max_hw_sectors;
    u32 stripe_size;
    u16 oncs;
    atomic_t abort_limit;
    u8 event_limit;
    u8 vwc;
    u32 vs;
    bool subsystem;
    unsigned long quirks;
};


/* Linux obama 4.4.21jonas #26 SMP Wed Dec 7 11:15:24 CET 2016 x86_64 x86_64 x86_64 GNU/Linux */
struct nvme_ns {
    struct list_head list;

    struct nvme_ctrl *ctrl;
    struct request_queue *queue;
    struct gendisk *disk;
    struct kref kref;

    u8 eui[8];
    u8 uuid[16];

    unsigned ns_id;
    int lba_shift;
    u16 ms;
    bool ext;
    u8 pi_type;
    int type;
    unsigned long flags;

    u64 mode_select_num_blocks;
    u32 mode_select_block_len;
};


/* Linux obama 4.4.21jonas #26 SMP Wed Dec 7 11:15:24 CET 2016 x86_64 x86_64 x86_64 GNU/Linux */
struct nvme_dev {
    struct nvme_queue **queues;
    struct blk_mq_tag_set tagset;
    struct blk_mq_tag_set admin_tagset;
    u32 __iomem *dbs;
    struct device *dev;
    struct dma_pool *prp_page_pool;
    struct dma_pool *prp_small_pool;
    unsigned queue_count;
    unsigned online_queues;
    unsigned max_qid;
    int q_depth;
    u32 db_stride;
    struct msix_entry *entry;
    void __iomem *bar;
    struct work_struct reset_work;
    struct work_struct scan_work;
    struct work_struct remove_work;
    struct work_struct async_work;
    struct timer_list watchdog_timer;
    struct mutex shutdown_lock;
    bool subsystem;
    void __iomem *cmb;
    dma_addr_t cmb_dma_addr;
    u64 cmb_size;
    u32 cmbsz;
    unsigned long flags;

    struct nvme_ctrl ctrl;
    struct completion ioq_wait;
};


static inline struct nvme_dev* ctrl_to_dev(struct nvme_ctrl* ctrl)
{
    return container_of(ctrl, struct nvme_dev, ctrl);
}


static inline struct nvme_dev* ns_to_dev(struct nvme_ns* ns)
{
    return ctrl_to_dev(ns->ctrl);
}


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


