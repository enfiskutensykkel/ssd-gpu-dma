#include "ctrl_ref.h"
#include "ctrl_dev.h"
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <asm/io.h>
#include <asm/errno.h>
#include <asm/page.h>

#define DRIVER_NAME         "disnvme"
#define PCI_CLASS_NVME      0x010802
#define PCI_CLASS_NVME_MASK 0xffffff

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Set up DMA mappings for userspace buffers");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION("0.2");


/* Define a filter for selecting devices we are interested in */
static const struct pci_device_id id_table[] = 
{
    { PCI_DEVICE_CLASS(PCI_CLASS_NVME, PCI_CLASS_NVME_MASK) },
    { 0 }
};


/* Reference to the first character device */
static dev_t dev_first;


/* Device class */
static struct class* dev_class;


/* Array of devices */
static struct ctrl_dev* ctrl_devs = NULL;


/* Array of controller references */
static struct ctrl_ref* ctrl_refs = NULL;


/* Number of devices */
static int num_ctrl_devs = 8;
module_param(num_ctrl_devs, int, 0);
MODULE_PARM_DESC(num_ctrl_devs, "Number of controller device slots");

/* Number of controller references */
static long num_ctrl_refs = 48;
module_param(num_ctrl_refs, long, 0);
MODULE_PARM_DESC(num_ctrl_refs, "Number of controller reference slots");

/* Number of pages per mapping */
static long max_pages_per_map = 0x8000;
module_param(max_pages_per_map, long, 0);
MODULE_PARM_DESC(max_pages_per_map, "Maximum number of pages per mapping");



static struct ctrl_ref* find_ref(struct ctrl_dev* dev, struct task_struct* owner)
{
    long i;
    struct ctrl_ref* ref;

    for (i = 0; i < num_ctrl_refs; ++i)
    {
        ref = &ctrl_refs[i];

        if (ref->ctrl == dev && (owner == NULL || ref->owner == owner))
        {
            return ref;
        }
    }

    return NULL;
}


static struct ctrl_dev* find_dev_by_inode(struct inode* inode)
{
    int i;
    struct ctrl_dev* dev;

    for (i = 0; i < num_ctrl_devs; ++i)
    {
        dev = &ctrl_devs[i];

        if (inode->i_cdev == &dev->cdev)
        {
            if (ctrl_devs[i].pdev == NULL)
            {
                printk(KERN_CRIT "Controller device is removed but reference still exists!\n");
            }

            return dev;
        }
    }

    return NULL;
}


static struct ctrl_dev* find_dev_by_pdev(const struct pci_dev* pdev)
{
    int i;
    struct ctrl_dev* dev;

    for (i = 0; i < num_ctrl_devs; ++i)
    {
        dev = &ctrl_devs[i];

        if (dev->pdev == pdev)
        {
            return dev;
        }
    }

    return NULL;
}


static int ref_get(struct inode* inode, struct file* file)
{
    int i;
    struct ctrl_dev* dev;
    struct ctrl_ref* ref = NULL;

    // Find the character device in question
    dev = find_dev_by_inode(inode);
    if (dev == NULL)
    {
        printk(KERN_ALERT "Unknown character device!\n");
        return -EBADF;
    }

    // Do some sanity checking to ensure that controller is still around
    if (dev->pdev == NULL)
    {
        return -EBADF;
    }

    // Find available controller reference
    for (i = 0; i  < num_ctrl_refs; ++i)
    {
        ref = ctrl_ref_get(&ctrl_refs[i], dev);
        if (ref != NULL)
        {
            return 0;
        }
    }

    printk(KERN_WARNING "No available controller reference slots\n");
    return -ENOSPC;
}


static int ref_put(struct inode* inode, struct file* file)
{
    struct ctrl_dev* dev;
    struct ctrl_ref* ref;

    dev = find_dev_by_inode(inode);
    if (dev == NULL)
    {
        printk(KERN_CRIT "Unknown controller device!\n");
        return -EBADF;
    }
    
    ref = find_ref(dev, current);
    if (ref == NULL)
    {
        printk(KERN_ERR "No controller references found but device exists!\n");
        return -EACCES;
    }

    ctrl_ref_put(ref);
    return 0;
}


static long ref_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
    return 0;
}


static int ref_mmap(struct file* file, struct vm_area_struct* vma)
{
    struct ctrl_dev* dev;
    struct ctrl_ref* ref;

    dev = find_dev_by_inode(file->f_inode);
    if (dev == NULL)
    {
        printk(KERN_CRIT "Unknown controller device!\n");
        return -EBADF;
    }

    if (dev->pdev == NULL)
    {
        printk(KERN_ALERT "Controller device exists but PCI device is removed\n");
        return -EAGAIN;
    }

    ref = find_ref(dev, current);
    if (ref == NULL)
    {
        printk(KERN_ERR "No controller references found but device exists!\n");
        return -EACCES;
    }

    if (vma->vm_end - vma->vm_start > pci_resource_len(dev->pdev, 0))
    {
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    return vm_iomap_memory(vma, pci_resource_start(dev->pdev, 0), vma->vm_end - vma->vm_start);
}


/* Define file operations for device file */
static const struct file_operations dev_fops = 
{
    .owner = THIS_MODULE,
    .open = ref_get,
    .release = ref_put,
    .unlocked_ioctl = ref_ioctl,
    .mmap = ref_mmap,
};


static int add_pci_dev(struct pci_dev* pdev, const struct pci_device_id* id)
{
    int i;
    int err;
    struct ctrl_dev* dev = NULL;

    printk(KERN_NOTICE "Adding controller device: %02x:%02x.%1x", 
            pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));

    // Find free slot
    for (i = 0; i < num_ctrl_devs; ++i)
    {
        dev = ctrl_dev_get(&ctrl_devs[i], pdev);
        if (dev != NULL)
        {
            break;
        }
    }

    if (dev == NULL)
    {
        printk(KERN_WARNING "No available controller device slots\n");
        return -ENOSPC;
    }

    err = pci_request_region(pdev, 0, DRIVER_NAME);
    if (err != 0)
    {
        ctrl_dev_put(dev);
        return err;
    }

    // Enable PCI device
    err = pci_enable_device(pdev);
    if (err < 0)
    {
        pci_release_region(pdev, 0);
        ctrl_dev_put(dev);
        printk(KERN_ERR "Failed to enable PCI device\n");
        return err;
    }

    // Create character device file
    err = ctrl_dev_chrdev_create(dev, &dev_fops);
    if (err != 0)
    {
        pci_release_region(pdev, 0);
        pci_disable_device(pdev);
        ctrl_dev_put(dev);
        return err;
    }

    // Enable DMA
    pci_set_master(pdev);
    return 0;
}


static void remove_pci_dev(struct pci_dev* pdev)
{
    struct ctrl_ref* ref;
    struct ctrl_dev* dev;

    if (pdev == NULL)
    {
        printk(KERN_WARNING "Remove controller device invoked with NULL\n");
        return;
    }

    // Disable PCI device
    pci_clear_master(pdev);
    pci_disable_device(pdev);
    pci_release_region(pdev, 0);

    // Find controller device in question
    dev = find_dev_by_pdev(pdev);
    if (dev == NULL)
    {
        printk(KERN_ALERT "Attempting to remove unknown PCI device: %02x:%02x.%1x\n",
                pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
        return;
    }

    ref = find_ref(dev, NULL);
    if (ref != NULL) 
    {
        printk(KERN_CRIT "Controller device is still referenced by at least one process: %02x:%02x.%1x\n", 
                pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
    }

    // Remove character device
    ctrl_dev_put(dev);

    printk(KERN_NOTICE "Controller device removed: %02x:%02x.%1x", 
            pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
}


/* Define driver operations we support */
static struct pci_driver driver = 
{
    .name = DRIVER_NAME,
    .id_table = id_table,
    .probe = add_pci_dev,
    .remove = remove_pci_dev,
};


static int __init disnvme_entry(void)
{
    int err;
    int i;

    // Allocate array of device handles
    ctrl_devs = kcalloc(num_ctrl_devs, sizeof(struct ctrl_dev), GFP_KERNEL);
    if (ctrl_devs == NULL)
    {
        printk(KERN_CRIT "Failed to allocate controller device handles\n");
        return -ENOMEM;
    }

    // Allocate array of controller reference pointers
    ctrl_refs = kcalloc(num_ctrl_refs, sizeof(struct ctrl_ref), GFP_KERNEL);
    if (ctrl_refs == NULL)
    {
        kfree(ctrl_devs);
        printk(KERN_CRIT "Failed to allocate controller reference slots\n");
        return -ENOMEM;
    }

    // Set up character device creation
    err = alloc_chrdev_region(&dev_first, 0, num_ctrl_devs, DRIVER_NAME);
    if (err < 0)
    {
        kfree(ctrl_refs);
        kfree(ctrl_devs);
        printk(KERN_CRIT "Failed to allocate chrdev region\n");
        return err;
    }

    dev_class = class_create(THIS_MODULE, DRIVER_NAME);
    if (IS_ERR(dev_class))
    {
        unregister_chrdev_region(dev_first, num_ctrl_devs);
        kfree(ctrl_refs);
        kfree(ctrl_devs);
        printk(KERN_CRIT "Failed to create chrdev class\n");
        return PTR_ERR(dev_class);
    }

    // Reset all controller device handles
    for (i = 0; i < num_ctrl_devs; ++i)
    {
        ctrl_dev_reset(&ctrl_devs[i], dev_first, i, dev_class, DRIVER_NAME);
    }

    // Register as PCI driver
    err = pci_register_driver(&driver);
    if (err != 0)
    {
        class_destroy(dev_class);
        unregister_chrdev_region(dev_first, num_ctrl_devs);
        kfree(ctrl_refs);
        kfree(ctrl_devs);
        printk(KERN_CRIT "Failed to register as PCI driver\n");
        return err;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded (num_ctrl_devs=%d num_ctrl_refs=%ld max_pages_per_map=%ld)\n",
            num_ctrl_devs, num_ctrl_refs, max_pages_per_map);

    return 0;
}
module_init(disnvme_entry);


static void __exit disnvme_exit(void)
{
    pci_unregister_driver(&driver);
    class_destroy(dev_class);
    unregister_chrdev_region(dev_first, num_ctrl_devs);

    // FIXME: Should we loop through devs and refs here?

    kfree(ctrl_refs);
    kfree(ctrl_devs);

    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}
module_exit(disnvme_exit);

