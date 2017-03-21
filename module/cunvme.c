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


/* Some handy constants */
#define RAM_BOUND_SIZE      PAGE_SIZE
#define RAM_BOUND_MASK      PAGE_MASK
#define GPU_BOUND_SIZE      (1UL << 16)
#define GPU_BOUND_MASK      ~(GPU_BOUND_SIZE - 1)


/* Some general information */
MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Page-lock userspace buffers and retrieve physical addresses of their pages");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION(CUNVME_VERSION);


/* Describes a handle to bound memory */
struct page_handle
{
    struct task_struct*         owner;          /* user process that owns the pages */
    struct page**               ram_pages;      /* array of pointers to pinned pages */
    struct vm_area_struct**     vmas;           /* virtual memory mapping stuff */
    nvidia_p2p_page_table_t*    gpu_pages;      /* gpu pages in question */
    unsigned long long          vaddr_start;    /* start address of the pages */
    long                        num_pages;      /* number of pages pinned */
};


/* Spinlock that ensures atomic access to the page handles */
static spinlock_t page_handles_lock; 


/* List of page handles */
static struct page_handle* page_handles = NULL;


/* Maximum number of page handles allowed */
static long num_page_handles = 0;
module_param(num_page_handles, long, 0);
MODULE_PARM_DESC(num_page_handles, "Number of available handles to pinned memory regions");


/* Maximum number of user pages per handle */
static long max_ram_pages = 0;
module_param(max_ram_pages, long, 0);
MODULE_PARM_DESC(max_ram_pages, "Maximum number of RAM pages per handle");


/* Maximum number of GPU pages per handle */
static long max_gpu_pages = 0;
module_param(max_gpu_pages, long, 0);
MODULE_PARM_DESC(max_gpu_pages, "Maximum number of GPU pages per handle");


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


/* 
 * Free callback used when Nvidia driver must 
 * forcefully take back pinned memory 
 */
static void free_callback(void* handlep)
{
    struct page_handle* handle = (struct page_handle*) handlep;

    if (handle->owner != NULL && handle->gpu_pages != NULL)
    {
        nvidia_p2p_free_page_table(handle->gpu_pages);
        handle->gpu_pages = NULL;
    }
}


/* Get a free page handle */
static struct page_handle* get_page_handle(void)
{
    long i;

    for (i = 0; i < num_page_handles; ++i)
    {
        spin_lock(&page_handles_lock);
        if (page_handles[i].owner == NULL)
        {
            page_handles[i].owner = current;
            spin_unlock(&page_handles_lock);

            page_handles[i].num_pages = 0;
            page_handles[i].ram_pages = NULL;
            page_handles[i].vmas = NULL;
            page_handles[i].gpu_pages = NULL;
            return &page_handles[i];
        }
        spin_unlock(&page_handles_lock);
    }

    return NULL;
}


/* Release a page handle */
static int put_page_handle(struct page_handle* handle)
{
    long curr_page;

    if (handle != NULL)
    {
        if (handle->ram_pages != NULL)
        {
            for (curr_page = 0; curr_page < handle->num_pages; ++curr_page)
            {
                put_page(handle->ram_pages[curr_page]);
            }
            
            kfree(handle->ram_pages);
            handle->ram_pages = NULL;
        }

        if (handle->vmas != NULL)
        {
            kfree(handle->vmas);
            handle->vmas = NULL;
        }

        if (handle->gpu_pages != NULL)
        {
            nvidia_p2p_put_pages(0, 0, handle->vaddr_start, handle->gpu_pages);
            handle->gpu_pages = NULL;
        }

        handle->num_pages = 0;
        handle->owner = NULL;
        return 0;
    }

    return -EBADF;
}


static long unpin_memory(void __user* requestp)
{
    struct cunvme_unpin request;

    if (copy_from_user(&request, requestp, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        return -EIO;
    }

    if (request.handle < 0 || request.handle >= num_page_handles || request.handle == CUNVME_NO_HANDLE)
    {
        printk(KERN_WARNING "Invalid kernel handle from %d: %ld\n", current->pid, request.handle);
        return -EBADF;
    }

    spin_lock(&page_handles_lock);
    if (page_handles[request.handle].owner != current)
    {
        spin_unlock(&page_handles_lock);
        printk(KERN_WARNING "Attempted to release handle not owned by %d\n", current->pid);
        return -EACCES;
    }
    spin_unlock(&page_handles_lock);

    return put_page_handle(&page_handles[request.handle]);
}


static long pin_gpu_memory(void __user* requestp)
{
    struct cunvme_pin request;
    struct page_handle* handle;
    int retval = 0;
    long i;
    unsigned long long* ptr;
    unsigned long long addr;

    handle = get_page_handle();
    if (handle == NULL)
    {
        printk(KERN_WARNING "Maximum number of page handles reached\n");
        retval = -ENOMEM;
        goto out;
    }

    if (copy_from_user(&request, requestp, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        retval = -EIO;
        goto out;
    }

    request.handle = handle - page_handles;
    handle->vaddr_start = request.virt_addr & GPU_BOUND_MASK;
    handle->num_pages = 0;
    
    if (request.num_pages <= 0 || request.num_pages > max_gpu_pages)
    {
        printk(KERN_WARNING "Number of GPU pages requested by %d exceeds limit: %ld\n", current->pid, request.num_pages);
        retval = -EINVAL;
        goto out;
    }

    // Pin pages and get handles
    retval = nvidia_p2p_get_pages(0, 0, handle->vaddr_start, GPU_BOUND_SIZE * request.num_pages, &handle->gpu_pages, free_callback, handle);
    if (retval != 0)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", retval);
        goto out;
    }

    // Update number of pages to actual number of pages
    handle->num_pages = handle->gpu_pages->entries;
    request.num_pages = handle->num_pages;

    // Walk page table and set bus addresses
    ptr = (unsigned long long*) (((unsigned char*) requestp) + offsetof(struct cunvme_pin, bus_addr));
    for (i = 0; i < handle->num_pages; ++i)
    {
        addr = handle->gpu_pages->pages[i]->physical_address;
        copy_to_user(ptr + i, &addr, sizeof(addr));
    }

out:
    if (retval != 0)
    {
        request.handle = CUNVME_NO_HANDLE;
        put_page_handle(handle);
    }

    if (copy_to_user(requestp, &request, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to write back result to userspace\n");
        retval = -EIO;
    }

    return retval;
}


static long pin_user_pages(void __user* requestp)
{
    struct cunvme_pin request;
    struct page_handle* handle;
    long retval = 0;
    long i;
    unsigned long long* ptr;
    unsigned long long addr;

    handle = get_page_handle();
    if (handle == NULL)
    {
        printk(KERN_WARNING "Maximum number of page handles reached\n");
        retval = -ENOMEM;
        goto out;
    }

    if (copy_from_user(&request, requestp, sizeof(request)) != 0)
    {
        printk(KERN_ERR "Failed to read userspace request from %d\n", current->pid);
        retval = -EIO;
        goto out;
    }

    request.handle = handle - page_handles;
    handle->vaddr_start = request.virt_addr & RAM_BOUND_MASK;
    handle->num_pages = 0;
    
    if (request.num_pages <= 0 || request.num_pages > max_ram_pages)
    {
        printk(KERN_WARNING "Number of RAM pages requested by %d exceeds limit: %ld\n", current->pid, request.num_pages);
        retval = -EINVAL;
        goto out;
    }

    handle->ram_pages = (struct page**) kcalloc(request.num_pages, sizeof(struct page*), GFP_KERNEL);
    if (handle->ram_pages == NULL)
    {
        printk(KERN_ERR "Failed to allocate user page table\n");
        retval = -ENOMEM;
        goto out;
    }

    handle->vmas = (struct vm_area_struct**) kcalloc(request.num_pages, sizeof(struct vm_area_struct*), GFP_KERNEL);
    if (handle->vmas == NULL)
    {
        printk(KERN_ERR "Failed to allocate user page table\n");
        retval = -ENOMEM;
        goto out;
    }

    //#if (LINUX_VERSION_CODE <= KERNEL_VERSION(4, 4, 57))
    retval = get_user_pages(current, current->mm, handle->vaddr_start, request.num_pages, 1, 0, handle->ram_pages, handle->vmas);
    //#endif
    
    if (retval <= 0)
    {
        printk(KERN_ERR "get_user_pages() failed with error code: %lu\n", retval);
        goto out;
    }

    handle->num_pages = retval;
    request.num_pages = retval;
    retval = 0;

    // Walk page table and set bus addresses
    ptr = (unsigned long long*) (((unsigned char*) requestp) + offsetof(struct cunvme_pin, bus_addr));
    for (i = 0; i < handle->num_pages; ++i)
    {
        addr = page_to_phys(handle->ram_pages[i]);
        copy_to_user(ptr + i, &addr, sizeof(addr));
    }
    
out:
    if (retval != 0)
    {
        request.handle = CUNVME_NO_HANDLE;
        put_page_handle(handle);
    }

    if (copy_to_user(requestp, &request, sizeof(request)) != 0)
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
        case CUNVME_PIN_RAM:
            retval = pin_user_pages((void __user*) arg);
            break;

        case CUNVME_PIN_GPU:
            retval = pin_gpu_memory((void __user*) arg);
            break;

        case CUNVME_UNPIN:
            retval = unpin_memory((void __user*) arg);
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

    for (i = in_use = 0; i < num_page_handles; ++i)
    {
        if (page_handles[i].owner == current)
        {
            put_page_handle(&page_handles[i]);
            ++in_use;
        }
    }

    if (in_use > 0)
    {
        printk(KERN_INFO "%ld handles were still in use by process %d\n", in_use, current->pid);
    }

    return 0;
}


static int __init cunvme_entry(void)
{
    long i;

    spin_lock_init(&page_handles_lock);

    page_handles = kcalloc(num_page_handles, sizeof(struct page_handle), GFP_KERNEL);
    if (page_handles == NULL)
    {
        printk(KERN_ERR "Failed to allocate page handles\n");
        return -ENOMEM;
    }

    for (i = 0; i < num_page_handles; ++i)
    {
        page_handles[i].owner = NULL;
    }
    
    ioctl_file = proc_create(CUNVME_FILENAME, 0, NULL, &ioctl_fops);
    if (ioctl_file == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", CUNVME_FILENAME);
        kfree(page_handles);
        return -ENOMEM;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded (num_page_handles=%ld max_ram_pages=%lu max_gpu_pages=%lu)\n",
            num_page_handles, max_ram_pages, max_gpu_pages);

    return 0;
}
module_init(cunvme_entry);


static void __exit cunvme_exit(void)
{
    long i;
    long count = 0;

    proc_remove(ioctl_file);

    for (i = 0; i < num_page_handles; ++i)
    {
        if (page_handles[i].owner != NULL)
        {
            ++count;
            put_page_handle(&page_handles[i]); // FIXME not sure if this works, but should never happen
        }
    }

    kfree(page_handles);

    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");

    if (count > 0)
    {
        printk(KERN_CRIT "%ld handles were still in use!!\n", count);
    }
}
module_exit(cunvme_exit);

