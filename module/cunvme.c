#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <asm/io.h>
#include <linux/mm.h>
#include <linux/spinlock.h>
#include <cunvme_ioctl.h>


/* Hold pinned pages in a data structure */
struct page_descriptor
{
    
};


/* Maximum number of pinned pages allowed */
static long num_user_pages = 0;
module_param(num_user_pages, long, 64);
MODULE_PARM_DESC(num_user_pages, "Maximum number of pinned pages");


/* Pinned pages */
static struct page_descriptor* user_pages = NULL;


/* Pointer to the current page entry */
static long current_page = 0;


/* Ensure atomic access to page descriptor table */
static spinlock_t lock = __SPIN_LOCK_UNLOCKED(lock);


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


static long pin_user_page(struct cunvme_pin_page __user* request_ptr)
{
    struct cunvme_pin_page request;
    int retval = 0;
    long handle = CUNVME_NO_HANDLE;

    if (copy_from_user(&request, request_ptr, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request\n");
        return -EIO;
    }
    
    spin_lock(&lock);
    retval = -ENOMEM;
    if (current_page < num_user_pages)
    {
        retval = 0;
        handle = current_page++;
    }
    spin_unlock(&lock);

    if (retval == 0)
    {
        //current, current->mm
    }

    request.handle = handle;
    if (copy_to_user(request_ptr, &request, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to write back result to userspace\n");
        retval = -EIO;
    }

    return retval;
}


/* Entry point to kernel module */
static long handle_request(struct file* ioctl_file, unsigned int cmd, unsigned long arg)
{
    long retval;

    switch (cmd)
    {
        case CUNVME_PIN:
            retval = pin_user_page((struct cunvme_pin_page __user*) arg);
            break;

        default:
            retval = -EINVAL;
            break;
    }

    return retval;
}


/* Release pinned memory pages */
static int release_file(struct inode* inode, struct file* file)
{
    return 0;
}


static int __init unvme_entry(void)
{
    user_pages = kcalloc(num_user_pages, sizeof(struct page_descriptor), GFP_KERNEL);
    if (user_pages == NULL)
    {
        printk(KERN_ERR "Failed to allocate user page descriptor table\n");
        return -ENOMEM;
    }

    ioctl_file = proc_create(CUNVME_FILE, 0, NULL, &ioctl_fops);
    if (ioctl_file == NULL)
    {
        kfree(user_pages);
        printk(KERN_ERR "Failed to create proc file: %s\n", CUNVME_FILE);
        return -ENOMEM;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded\n");
    return 0;
}


static void __exit unvme_exit(void)
{
    proc_remove(ioctl_file);
    kfree(user_pages);
    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}


module_init(unvme_entry);
module_exit(unvme_exit);

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Stub module to page-lock memory and retrieve physical addresses");
MODULE_LICENSE("GPL");
MODULE_VERSION(CUNVME_VERSION);
