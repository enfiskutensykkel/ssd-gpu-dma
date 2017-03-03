#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <asm/io.h>
#include <linux/mm.h>
#include <cunvme_ioctl.h>


static unsigned int memory_desc_idx = 0;


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


static long pin_memory(struct cunvme_request __user* request)
{
    struct cunvme_request r;
    copy_from_user(&r, request, sizeof(r));
    
    unsigned int idx = memory_desc_idx++;
    printk(KERN_INFO "%u\n", idx);

    //get_user_pages(current, current->mm, r.vaddr, r.size / PAGE_SIZE, 0, &(pages[i].pages), NULL);
    
    r.handle = idx;
    r.paddr = 0;

    copy_to_user(request, &r, sizeof(r));
    return 0;
}


/* Entry point to kernel module */
static long handle_request(struct file* ioctl_file, unsigned int cmd, unsigned long arg)
{
    long retval;

    switch (cmd)
    {
        case CUNVME_PIN:
            retval = pin_memory((struct cunvme_request __user*) arg);
            break;

        case CUNVME_UNPIN:
            retval = 0;
            break;

        case CUNVME_QUERY:
            retval = 0;
            break;

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
MODULE_DESCRIPTION("Stub module to page-lock memory and retrieve physical addresses");
MODULE_LICENSE("GPL");
MODULE_VERSION(CUNVME_VERSION);
