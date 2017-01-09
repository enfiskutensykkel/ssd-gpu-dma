#include <linux/module.h>

MODULE_AUTHOR("Jonas Markussen");
MODULE_DESCRIPTION("Use SSD DMA to transfer across PCIe NTB");
MODULE_LICENSE("GPL");
MODULE_VERSION("0.1"); // FIXME: Create this with define in Makefile


static int __init ssd_dma_entry(void)
{
    printk(KERN_INFO KBUILD_MODNAME " loaded\n");
    return 0;
}
module_init(ssd_dma_entry);


static void __exit ssd_dma_exit(void)
{
    printk(KERN_INFO KBUILD_MODNAME " unloaded\n");
}
module_exit(ssd_dma_exit);
