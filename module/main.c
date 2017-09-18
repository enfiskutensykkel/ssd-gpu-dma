#include "ctrl_ref.h"
#include "ctrl_dev.h"
#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/moduleparam.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>

#define DRIVER_NAME         "disnvme"
#define PCI_CLASS_NVME      0x010802
#define PCI_CLASS_NVME_MASK 0xffffff

MODULE_AUTHOR("Jonas Markussen <jonassm@simula.no>");
MODULE_DESCRIPTION("Set up DMA mappings for userspace buffers");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION("0.2");


/* Reference to the first character device */
static dev_t dev_first;


/* Device class */
static struct class* dev_class;


/* Array of devices */
static struct ctrl_dev* ctrl_devs = NULL;


/* Array of pointes to controller references */
static struct ctrl_ref** ctrl_refs = NULL;


/* Number of devices */
static int num_ctrl_devs = 8;
module_param(num_ctrl_devs, int, 0);
MODULE_PARM_DESC(num_ctrl_devs, "Maximum number of NVMe controllers supported");

/* Number of controller references */
static long num_ctrl_refs = 48;
module_param(num_ctrl_refs, long, 0);
MODULE_PARM_DESC(num_ctrl_refs, "Maximum number of controller references");

/* Number of pages per mapping */
static long pages_per_map = 0x8000;
module_param(pages_per_map, long, 0);
MODULE_PARM_DESC(pages_per_map, "Maximum number of pages per mapping");



/* Define file operations for device file */
static const struct file_operations dev_fops = 
{
    .owner = THIS_MODULE,
    .unlocked_ioctl = NULL,
    .mmap = NULL,
    .release = NULL,
};


/* Define a filter for selecting devices we are interested in */
static const struct pci_device_id id_table[] = 
{
    { PCI_DEVICE_CLASS(PCI_CLASS_NVME, PCI_CLASS_NVME_MASK) },
    { 0 }
};


static int add_pci_dev(struct pci_dev* pdev, const struct pci_device_id* id)
{
    int i;
    int err;

    printk(KERN_NOTICE "Adding device %02x:%02x.%1x", 
            pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));

    for (i = 0; i < num_ctrl_devs; ++i)
    {
        if (ctrl_devs[i].ctrl_pdev == NULL)
        {
            break;
        }
    }

    if (i == num_ctrl_devs)
    {
        printk(KERN_WARNING "No more available devices\n");
        return -ENOSPC;
    }

    ctrl_devs[i].ctrl_pdev = pdev;

    err = ctrl_dev_create(&ctrl_devs[i], i, dev_first, dev_class, &dev_fops, DRIVER_NAME);
    if (err != 0)
    {
        ctrl_devs[i].ctrl_pdev = NULL;
        return err;
    }

    return 0;
}


static void remove_pci_dev(struct pci_dev* pdev)
{
    int i;

    if (pdev == NULL)
    {
        printk(KERN_WARNING "Remove device invoked with NULL\n");
        return;
    }

    for (i = 0; i < num_ctrl_devs; ++i)
    {
        if (ctrl_devs[i].ctrl_pdev == pdev)
        {
            break;
        }
    }

    if (i == num_ctrl_devs)
    {
        printk(KERN_ALERT "Remove device invoked with unknown PCI device!\n");
        return;
    }

    ctrl_dev_remove(&ctrl_devs[i]);
    ctrl_devs[i].ctrl_pdev = NULL;

    printk(KERN_NOTICE "Removing device %02x:%02x.%1x", 
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


//   // pci_bus_addr_t b = pci_bus_address(pdev, bar);
//

static int __init disnvme_entry(void)
{
    int err;

    /* Allocate array of device handles */
    ctrl_devs = kcalloc(num_ctrl_devs, sizeof(struct ctrl_dev), GFP_KERNEL);
    if (ctrl_devs == NULL)
    {
        printk(KERN_CRIT "Failed to allocate device handles\n");
        return -ENOMEM;
    }

    /* Allocate array of controller reference pointers */
    ctrl_refs = kcalloc(num_ctrl_refs, sizeof(struct ctrl_ref*), GFP_KERNEL);
    if (ctrl_refs == NULL)
    {
        kfree(ctrl_devs);
        printk(KERN_CRIT "Failed to allocate device reference slots\n");
        return -ENOMEM;
    }

    /* Set up character device creation */
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

    /* Register as PCI driver */
    err = pci_register_driver(&driver);
    if (err != 0)
    {
        class_destroy(dev_class);
        unregister_chrdev_region(dev_first, num_ctrl_devs);
        kfree(ctrl_refs);
        kfree(ctrl_devs);
        printk(KERN_CRIT"Failed to register as PCI driver\n");
        return err;
    }

    printk(KERN_DEBUG KBUILD_MODNAME " loaded (num_ctrl_devs=%d num_ctrl_refs=%ld pages_per_map=%ld)\n",
            num_ctrl_devs, num_ctrl_refs, pages_per_map);

    return 0;
}
module_init(disnvme_entry);


static void __exit disnvme_exit(void)
{
    pci_unregister_driver(&driver);
    class_destroy(dev_class);
    unregister_chrdev_region(dev_first, num_ctrl_devs);

    /* TODO: sanity checking */

    kfree(ctrl_refs);
    kfree(ctrl_devs);

    printk(KERN_DEBUG KBUILD_MODNAME " unloaded\n");
}
module_exit(disnvme_exit);

