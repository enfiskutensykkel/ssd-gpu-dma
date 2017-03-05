#include <linux/module.h>
//#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
//#include <linux/slab.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <asm/io.h>
//#include <linux/spinlock.h>
//#include <linux/sched.h>
#include <cunvme_ioctl.h>
#include <linux/mm.h>


//static long num_user_pages = 0;
//module_param(num_user_pages, long, 0);
//MODULE_PARM_DESC(num_user_pages, "Maximum number of pinned pages");

//static spinlock_t lock = __SPIN_LOCK_UNLOCKED(lock);


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


static long lookup_user_page(struct cunvme_virt_to_phys __user* request_ptr)
{
    struct cunvme_virt_to_phys request;
    int retval = 0;
    struct page* page;
    dma_addr_t ph_addr;

    if (copy_from_user(&request, request_ptr, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request\n");
        return -EIO;
    }
    
    // TODO call get_user_pages instead and use that for page-locking?
    //retval = get_user_pages(current, current->mm, request.vaddr, 1, 1, 0, &pd->page, &pd->vm_area);
    
    page = virt_to_page(request.vaddr);
    ph_addr = page_to_phys(page);

    printk(KERN_DEBUG "vaddr=%llx paddr=%llx\n", request.vaddr, ph_addr);

    request.paddr = ph_addr;

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
        case CUNVME_VIRT_TO_PHYS:
            retval = lookup_user_page((struct cunvme_virt_to_phys __user*) arg);
            break;

        default:
            printk(KERN_WARNING KBUILD_MODNAME ": got unknown request\n");
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


static int __init cunvme_entry(void)
{
    //user_pages = kcalloc(num_user_pages, sizeof(struct page_descriptor), GFP_KERNEL);
    
    ioctl_file = proc_create(CUNVME_FILE, 0, NULL, &ioctl_fops);
    if (ioctl_file == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", CUNVME_FILE);
        return -ENOMEM;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded\n");
    return 0;
}


static void __exit cunvme_exit(void)
{
    proc_remove(ioctl_file);
    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}


module_init(cunvme_entry);
module_exit(cunvme_exit);

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Stub module to page-lock memory and retrieve physical addresses");
MODULE_LICENSE("GPL");
MODULE_VERSION(CUNVME_VERSION);
