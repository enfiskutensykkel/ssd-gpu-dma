#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
#include <ssd_dma.h>

MODULE_AUTHOR("Jonas Markussen");
MODULE_DESCRIPTION("Use SSD DMA to transfer across PCIe NTB");
MODULE_LICENSE("GPL");
MODULE_VERSION(SSD_DMA_VERSION);


/* Kernel module entry point declaration */
static long ssd_dma_ioctl(struct file*, unsigned int, unsigned long);


/* Define file operations for doing ioctl */
static const struct file_operations ioctl_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = ssd_dma_ioctl,
};


/* Entry point for the kernel module */
static long ssd_dma_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
    long retval;

    switch (cmd)
    {
        case SSD_DMA_START_TRANSFER:
            retval = 0;
            printk(KERN_DEBUG "we're in!\n");
            break;

        default:
            retval = -EINVAL;
            break;
    }

    return retval;
}


static int __init ssd_dma_entry(void)
{
    if (proc_create(SSD_DMA_FILE_NAME, 0, NULL, &ioctl_fops) == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", SSD_DMA_FILE_NAME);
        return -ENOSPC;
    }

    printk(KERN_INFO KBUILD_MODNAME " loaded\n");
    return 0;
}
module_init(ssd_dma_entry);


static void __exit ssd_dma_exit(void)
{
    remove_proc_entry(SSD_DMA_FILE_NAME, NULL);
    printk(KERN_INFO KBUILD_MODNAME " unloaded\n");
}
module_exit(ssd_dma_exit);
