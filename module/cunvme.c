#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/version.h>
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
#include <nvidia/nv-p2p.h>

//#define PAGE_MASK ~((1 << PAGE_SHIFT) - 1)

#define GPU_BOUND_SHIFT     16
#define GPU_BOUND_SIZE      (1UL << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET    (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK      (~GPU_BOUND_OFFSET)


/* Describes a page of user memory */
struct user_page
{
    struct task_struct*         owner;      /* user process that owns the page */
    struct page*                page;       /* page in question */
    nvidia_p2p_page_table_t*    page_tbl;   /* gpu pages in question */
    struct vm_area_struct*      vma;        /* virtual memory mapping stuff */
    unsigned long long          virt_addr;  /* virtual address of the page */
    dma_addr_t                  phys_addr;  /* physical address of page */
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


/* Free callback */
static void free_gpu_page(void* data)
{
    struct user_page* up = (struct user_page*) data;

    if (up->page_tbl != NULL)
    {
        //nvidia_p2p_free_pages(up->page_tbl);
        nvidia_p2p_free_page_table(up->page_tbl);
    }

    up->page_tbl = NULL;
    up->owner = NULL;

    printk(KERN_DEBUG "Free'd GPU memory\n");
}


/* Get a free slot in the user page table */
static struct user_page* get_user_page_slot(void)
{
    long handle;

    for (handle = 0; handle < num_user_pages; ++handle)
    {
        spin_lock(&user_pages_lock);
        if (user_pages[handle].owner == NULL)
        {
            user_pages[handle].owner = current;
            user_pages[handle].page = NULL;
            spin_unlock(&user_pages_lock);
            return &user_pages[handle];
        }
        spin_unlock(&user_pages_lock);
    }

    return NULL;
}


/* Release slot in user page table */
static int put_user_page_slot(struct user_page* up)
{
    if (up->owner == current)
    {
        if (up->page != NULL)
        {
            put_page(up->page);
            up->page = NULL;
        }
        else
        {
            nvidia_p2p_put_pages(0, 0, up->virt_addr, up->page_tbl);
        }

        up->owner = NULL;
        return 0;
    }

    return -EBADF;
}


static long unpin_user_page(struct cunvme_unpin __user* requestp)
{
    struct cunvme_unpin request;

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

    return put_user_page_slot(&user_pages[request.handle]);
}


static long pin_user_page(struct cunvme_pin __user* request_ptr)
{
    struct cunvme_pin request;
    struct user_page* up = NULL;
    int retval = 0;
    long handle = CUNVME_NO_HANDLE;

    up = get_user_page_slot();
    if (up == NULL)
    {
        printk(KERN_WARNING "Out of available page slots\n");
        retval = -ENOMEM;
        goto out;
    }

    handle = up - user_pages;

    if (copy_from_user(&request, request_ptr, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        return -EIO;
    }

    if (request.gpu != CUNVME_NO_CUDA_DEVICE)
    {
        up->page = NULL;
        up->virt_addr = request.vaddr & GPU_BOUND_MASK;

        retval = nvidia_p2p_get_pages(0, 0, up->virt_addr, GPU_BOUND_SIZE, &up->page_tbl, free_gpu_page, up);
        if (retval != 0)
        {
            handle = CUNVME_NO_HANDLE;
            printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", retval);
            goto out;
        }

        up->phys_addr = (dma_addr_t) up->page_tbl->pages[0]->physical_address;
    }
    else
    {
        up->page_tbl = NULL;
        up->virt_addr = request.vaddr & PAGE_MASK;

        //#if (LINUX_VERSION_CODE <= KERNEL_VERSION(4, 4, 57))
        retval = get_user_pages(current, current->mm, up->virt_addr, 1, 1, 0, &up->page, &up->vma);
        //#endif
        if (retval != 1)
        {
            handle = CUNVME_NO_HANDLE;
            printk(KERN_ERR "Call to get_user_pages() failed: %d\n", retval);
            goto out;
        }
        up->phys_addr = page_to_phys(up->page);
    }

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
        if (user_pages[i].owner == current)
        {
            if (user_pages[i].page != NULL)
            {
                put_page(user_pages[i].page);
            }
            user_pages[i].owner = NULL;
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
        user_pages[i].owner = NULL;
        user_pages[i].page = NULL;
        user_pages[i].page_tbl = NULL;
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
        if (user_pages[i].owner != NULL)
        {
            ++count;
            if (user_pages[i].page != NULL)
            {
                put_page(user_pages[i].page);
            }
            user_pages[i].owner = NULL;
        }
    }

    kfree(user_pages);
    printk(KERN_INFO KBUILD_MODNAME " unloaded\n");

    if (count > 0)
    {
        printk(KERN_WARNING "%ld pages were still pinned\n", count);
    }
}


module_init(cunvme_entry);
module_exit(cunvme_exit);

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Stub module to page-lock memory and retrieve physical addresses");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION(CUNVME_VERSION);
