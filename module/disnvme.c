#include <linux/module.h>
#include <linux/types.h>
#ifdef _CUDA
#include <nv-p2p.h>
#endif

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Set up DMA mappings for userspace buffers");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION("0.1");


static int __init disnvme_entry(void)
{
    printk(KERN_DEBUG KBUILD_MODNAME " loaded\n");
    return 0;
}
module_init(disnvme_entry);


static void __exit disnvme_exit(void)
{
    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}
module_exit(disnvme_exit);
