#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/file.h>
#include <asm/uaccess.h>
#include <asm/errno.h>


/* ioctl handler prototype */
static long handle_request(struct file*, unsigned int, unsigned long);


/* Release prototype */
static int release_file(struct inode*, struct file*);


/* Handle to ioctl file */
static struct proc_dir_entry* ioctl_file = NULL;


/* Define file operations for ioctl */
static const struct file_operations ioctl_fops = {
    .owner = THIS_MODULE,
    .release = release_file,
    .unlocked_ioctl = handle_request,
};


/* Entry point to kernel module */
static long handle_request(struct file* ioctl_file, unsigned int cmd, unsigned long arg)
{
    long retval;

    switch (cmd)
    {
        default:
            retval = -EINVAL;
            break;
    }

    return retval;
}



/* Release pinned GPU memory */
static int release_file(struct inode* inode, struct file* file)
{
    return 0;
}


static int __init unvme_entry(void)
{
    ioctl_file = proc_create(CUNVME_FILE, 0, NULL, &ioctl_fops);
    if (ioctl_file == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", CUNVME_FILE);
        return -ENOMEM;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded\n");
    return 0;
}


static void __exit unvme_exit(void)
{
    proc_remove(ioctl_file);
    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}


module_init(unvme_entry);
module_exit(unvme_exit);

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Stub module to pin GPU memory and retrieve physical addresses");
MODULE_LICENSE("GPL");
MODULE_VERSION(CUNVME_VERSION);
