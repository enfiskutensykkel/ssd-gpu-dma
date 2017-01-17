#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/proc_fs.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <ssd_dma.h>
#include "nvme.h"


/* Handler to NVMe driver and driver functions */
static struct module* mod_nvme = NULL;


/* IOCTL handler prototype */
static long proc_file_ioctl(struct file*, unsigned int, unsigned long);


/* Release prototype */
static int proc_file_release(struct inode*, struct file*);


/* Handle to proc file */
static struct proc_dir_entry* proc_file = NULL;


/* Define file operations for doing ioctl */
static const struct file_operations ioctl_fops = {
    .owner = THIS_MODULE,
    .release = proc_file_release,
    .unlocked_ioctl = proc_file_ioctl,
};


static int execute_dma_transfer(struct file* fp, dma_addr_t addr, size_t len, dma_vector_t* vec)
{
    
    return 0;
}


static int handle_start_transfer_request(dma_start_request_t __user* request_ptr)
{
    dma_start_request_t request;
    dma_vector_t* vector;
    struct file* file;
    int retval;

    if (copy_from_user(&request, request_ptr, sizeof(dma_start_request_t)) != 0)
    {
        printk(KERN_ERR "Failed to copy DMA request to kernel memory\n");
        return -EFAULT;
    }

    vector = kmalloc(sizeof(dma_vector_t) * request.vector_length, GFP_KERNEL);
    if (vector == NULL)
    {
        printk(KERN_ERR "Failed to allocate DMA vector\n");
        return -ENOMEM;
    }

    if (copy_from_user(vector, request_ptr->vector_elems, sizeof(dma_vector_t) * request.vector_length) != 0)
    {
        printk(KERN_ERR "Failed to copy DMA vector from user space\n");
        kfree(vector);
        return -EFAULT;
    }

    file = fget(request.file_desc);
    if (file == NULL)
    {
        printk(KERN_ERR "Failed to get file pointer for fd=%d\n", request.file_desc);
        return -EBADF;
    }

    printk(KERN_DEBUG "file size=%lli\n", i_size_read(file->f_inode));

    retval = execute_dma_transfer(file, request.io_address, request.vector_length, vector);

    fput(file);
    kfree(vector);
    return retval;
}


/* Entry point for the kernel module */
static long proc_file_ioctl(struct file* ioctl_file, unsigned int cmd, unsigned long arg)
{
    long retval;

    switch (cmd)
    {
        case SSD_DMA_START_TRANSFER:
            retval = handle_start_transfer_request((dma_start_request_t __user*) arg);
            break;

        default:
            retval = -EINVAL;
            break;
    }

    return retval;
}


/* Release any resources still held on to */
static int proc_file_release(struct inode* inode, struct file* file)
{
    return 0;
}


static int __init ssd_dma_entry(void)
{
    mutex_lock(&module_mutex);
    mod_nvme = find_module("nvme");
    if (!try_module_get(mod_nvme))
    {
        mutex_unlock(&module_mutex);
        printk(KERN_CRIT "Failed to ref\n");
        return -ENOENT;
    }

    mutex_unlock(&module_mutex);

    proc_file = proc_create(SSD_DMA_FILE_NAME, 0, NULL, &ioctl_fops);
    if (proc_file == NULL)
    {
        printk(KERN_ERR "Failed to create proc file: %s\n", SSD_DMA_FILE_NAME);
        return -ENOMEM;
    }

    printk(KERN_INFO KBUILD_MODNAME " loaded\n");
    return 0;
}


static void __exit ssd_dma_exit(void)
{
    proc_remove(proc_file);
    module_put(mod_nvme);

    printk(KERN_INFO KBUILD_MODNAME " unloaded\n");
}


module_init(ssd_dma_entry);
module_exit(ssd_dma_exit);
MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("SSD DMA across PCIe NTB");
MODULE_LICENSE("GPL");
MODULE_VERSION(SSD_DMA_VERSION);
