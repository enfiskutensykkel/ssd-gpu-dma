#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <asm/io.h>
#include <linux/spinlock.h>
#include <linux/sched.h>
#include <cunvme_ioctl.h>
#include <linux/mm.h>


struct user_page
{
    int                     used;
    struct task_struct*     owner;
    struct page*            page;
    struct vm_area_struct*  vma;
    unsigned long long      virt_addr;
    dma_addr_t              phys_addr;
};


static spinlock_t user_pages_lock; 


static struct user_page* user_pages = NULL;


static long num_user_pages = 0;
module_param(num_user_pages, long, 0);
MODULE_PARM_DESC(num_user_pages, "Maximum number of pinned pages");


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


static long unpin_user_page(struct cunvme_unpin __user* requestp)
{
    struct cunvme_unpin request;
    int retval = 0;
    struct user_page* up;

    if (copy_from_user(&request, requestp, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        return -EIO;
    }

    if (request.handle < 0 || request.handle >= num_user_pages)
    {
        printk(KERN_WARNING "Invalid kernel handle from %d: %ld\n", current->pid, request.handle);
        return -EBADF;
    }

    up = &user_pages[request.handle];

    spin_lock(&user_pages_lock);
    if (up->used && up->owner == current)
    {
        put_page(up->page);
        up->used = 0;
        up->owner = NULL;
    }
    else
    {
        retval = -EBADF;
    }
    spin_unlock(&user_pages_lock);

    return retval;
}


static long pin_user_page(struct cunvme_pin __user* request_ptr)
{
    struct cunvme_pin request;
    struct user_page* up = NULL;
    int retval = 0;
    long handle;

    for (handle = 0; handle < num_user_pages; ++handle)
    {
        spin_lock(&user_pages_lock);
        if (!user_pages[handle].used)
        {
            up = &user_pages[handle];
            up->used = 1;
            up->owner = current;
            spin_unlock(&user_pages_lock);
            break;
        }
        spin_unlock(&user_pages_lock);
    }

    if (up == NULL)
    {
        printk(KERN_WARNING "Out of available page slots\n");
        retval = -ENOMEM;
        goto out;
    }

    if (copy_from_user(&request, request_ptr, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        return -EIO;
    }

    retval = get_user_pages(current, current->mm, request.vaddr, 1, 1, 0, &up->page, &up->vma);
    if (retval != 1)
    {
        handle = CUNVME_NO_HANDLE;
        printk(KERN_ERR "Call to get_user_pages() failed: %d\n", retval);
        goto out;
    }

    up->virt_addr = request.vaddr;
    up->phys_addr = page_to_phys(up->page);
    request.paddr = up->phys_addr;

    printk(KERN_DEBUG "pid=%d vaddr=%llx paddr=%llx\n", current->pid, up->virt_addr, up->phys_addr);
    

out:
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
            retval = pin_user_page((struct cunvme_pin __user*) arg);
            break;

        case CUNVME_UNPIN:
            retval = unpin_user_page((struct cunvme_unpin __user*) arg);
            break;

        default:
            printk(KERN_WARNING "Unknown ioctl command from process %d: %u\n", current->pid, cmd);
            retval = -EINVAL;
            break;
    }

    return retval;
}


/* Release pinned memory pages */
static int release_file(struct inode* inode, struct file* file)
{
    long i;
    long in_use;

    printk(KERN_DEBUG "Cleaning up after process %d\n", current->pid);

    for (i = in_use = 0; i < num_user_pages; ++i)
    {
        spin_lock(&user_pages_lock);
        if (user_pages[i].used && user_pages[i].owner == current)
        {
            put_page(user_pages[i].page);
            user_pages[i].used = 0;
            ++in_use;
        }
        spin_unlock(&user_pages_lock);
    }

    if (in_use > 0)
    {
        printk(KERN_INFO "%ld pages were still pinned\n", in_use);
    }

    return 0;
}


static int __init cunvme_entry(void)
{
    long i;

    spin_lock_init(&user_pages_lock);
    user_pages = kcalloc(num_user_pages, sizeof(struct user_page), GFP_KERNEL);
    if (user_pages == NULL)
    {
        printk(KERN_ERR "Failed to allocate user page table\n");
        return -ENOMEM;
    }

    for (i = 0; i < num_user_pages; ++i)
    {
        user_pages[i].used = 0;
    }
    
    ioctl_file = proc_create(CUNVME_FILE, 0, NULL, &ioctl_fops);
    if (ioctl_file == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", CUNVME_FILE);
        return -ENOMEM;
    }

    printk(KERN_INFO KBUILD_MODNAME " loaded\n");
    return 0;
}


static void __exit cunvme_exit(void)
{
    long i;
    long count = 0;

    proc_remove(ioctl_file);

    for (i = 0; i < num_user_pages; ++i)
    {
        if (user_pages[i].used)
        {
            ++count;
            put_page(user_pages[i].page);
            user_pages[i].used = 0;
        }
    }

    kfree(user_pages);
    printk(KERN_INFO KBUILD_MODNAME " unloaded (%ld was still in use)\n", count);
}


module_init(cunvme_entry);
module_exit(cunvme_exit);

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Stub module to page-lock memory and retrieve physical addresses");
MODULE_LICENSE("GPL");
MODULE_VERSION(CUNVME_VERSION);
