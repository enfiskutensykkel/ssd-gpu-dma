#include "ioctl.h"
#include "list.h"
#include "ctrl.h"
#include "map.h"
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/err.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <asm/io.h>
#include <asm/errno.h>
#include <asm/page.h>

#define DRIVER_NAME         "libnvm helper"
#define PCI_CLASS_NVME      0x010802
#define PCI_CLASS_NVME_MASK 0xffffff


MODULE_AUTHOR("Jonas Markussen <jonassm@ifi.uio.no>");
MODULE_DESCRIPTION("Set up DMA mappings for userspace buffers");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_VERSION("0.3");


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


/* List of controller devices */
static struct list ctrl_list;


/* List of mapped host memory */
static struct list host_list;


/* List of mapped device memory */
static struct list device_list;


/* Number of devices */
static int num_ctrls = 8;
module_param(num_ctrls, int, 0);
MODULE_PARM_DESC(num_ctrls, "Number of controller devices");

static int curr_ctrls = 0;


static int mmap_registers(struct file* file, struct vm_area_struct* vma)
{
    struct ctrl* ctrl = NULL;

    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Unknown controller reference\n");
        return -EBADF;
    }

    if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0))
    {
        printk(KERN_WARNING "Invalid range size\n");
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    return vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0), vma->vm_end - vma->vm_start);
}



static long map_ioctl(struct file* file, unsigned int cmd, unsigned long arg)
{
    long retval = 0;
    struct ctrl* ctrl = NULL;
    struct nvm_ioctl_map request;
    struct map* map = NULL;
    u64 addr;

    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Unknown controller reference\n");
        return -EBADF;
    }

    switch (cmd)
    {
        case NVM_MAP_HOST_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                return -EFAULT;
            }

            map = map_userspace(&host_list, ctrl, request.vaddr_start, request.n_pages);

            if (!IS_ERR_OR_NULL(map))
            {
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t)))
                {
                    return -EFAULT;
                }
                retval = 0;
            }
            else 
            {
                retval = PTR_ERR(map);
            }
            break;

#ifdef _CUDA
        case NVM_MAP_DEVICE_MEMORY:
            if (copy_from_user(&request, (void __user*) arg, sizeof(request)))
            {
                return -EFAULT;
            }

            map = map_device_memory(&device_list, ctrl, request.vaddr_start, request.n_pages);

            if (!IS_ERR_OR_NULL(map))
            {
                if (copy_to_user((void __user*) request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t)))
                {
                    return -EFAULT;
                }
                retval = 0;
            }
            else 
            {
                retval = PTR_ERR(map);
            }
            break;
#endif

        case NVM_UNMAP_MEMORY:
            if (copy_from_user(&addr, (void __user*) arg, sizeof(u64)))
            {
                return -EFAULT;
            }

            map = map_find(&host_list, addr);
            if (map != NULL)
            {
                unmap_and_release(map);
                break;
            }

#ifdef _CUDA
            map = map_find(&device_list, addr);
            if (map != NULL)
            {
                unmap_and_release(map);
                break;
            }
#endif
            retval = -EINVAL;
            printk(KERN_WARNING "Mapping for address %llx not found\n", addr);
            break;

        default:
            printk(KERN_NOTICE "Unknown ioctl command from process %d: %u\n",
                    current->pid, cmd);
            retval = -EINVAL;
            break;
    }

    return retval;
}



/* Define file operations for device file */
static const struct file_operations dev_fops = 
{
    .owner = THIS_MODULE,
    .unlocked_ioctl = map_ioctl,
    .mmap = mmap_registers,
};


static int add_pci_dev(struct pci_dev* dev, const struct pci_device_id* id)
{
    int err;
    struct ctrl* ctrl = NULL;

    if (curr_ctrls >= num_ctrls)
    {
        printk(KERN_NOTICE "Maximum number of controller devices added\n");
        return 0;
    }

    printk(KERN_INFO "Adding controller device: %02x:%02x.%1x",
            dev->bus->number, PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));

    // Create controller reference
    ctrl = ctrl_get(&ctrl_list, dev_class, dev, curr_ctrls);
    if (IS_ERR(ctrl))
    {
        return PTR_ERR(ctrl);
    }

    // Get a reference to device memory
    err = pci_request_region(dev, 0, DRIVER_NAME);
    if (err != 0)
    {
        ctrl_put(ctrl);
        printk(KERN_ERR "Failed to get controller register memory\n");
        return err;
    }

    // Enable PCI device
    err = pci_enable_device(dev);
    if (err < 0)
    {
        pci_release_region(dev, 0);
        ctrl_put(ctrl);
        printk(KERN_ERR "Failed to enable controller\n");
        return err;
    }

    // Create character device file
    err = ctrl_chrdev_create(ctrl, dev_first, &dev_fops);
    if (err != 0)
    {
        pci_disable_device(dev);
        pci_release_region(dev, 0);
        ctrl_put(ctrl);
        return err;
    }

    // Enable DMA
    pci_set_master(dev);

    ++curr_ctrls;
    return 0;
}


static void remove_pci_dev(struct pci_dev* dev)
{
    struct ctrl* ctrl = NULL;

    if (dev == NULL)
    {
        printk(KERN_WARNING "Remove controller device was invoked with NULL\n");
        return;
    }

    --curr_ctrls;

    // Find controller reference
    ctrl = ctrl_find_by_pci_dev(&ctrl_list, dev);
    ctrl_put(ctrl);

    // Release device memory
    pci_release_region(dev, 0);

    // Disable PCI device
    pci_clear_master(dev);
    pci_disable_device(dev);

    printk(KERN_DEBUG "Controller device removed: %02x:%02x.%1x\n",
            dev->bus->number, PCI_SLOT(dev->devfn), PCI_FUNC(dev->devfn));
}


static unsigned long clear_map_list(struct list* list)
{
    unsigned long i = 0;
    struct list_node* ptr = list_next(&list->head);
    struct map* map;

    while (ptr != NULL)
    {
        map = container_of(ptr, struct map, list);
        unmap_and_release(map);
        ++i;

        ptr = list_next(&list->head);
    }

    return i;
}



/* Define driver operations we support */
static struct pci_driver driver = 
{
    .name = DRIVER_NAME,
    .id_table = id_table,
    .probe = add_pci_dev,
    .remove = remove_pci_dev,
};


static int __init libnvm_helper_entry(void)
{
    int err;

    list_init(&ctrl_list);
    list_init(&host_list);
    list_init(&device_list);

    // Set up character device creation
    err = alloc_chrdev_region(&dev_first, 0, num_ctrls, DRIVER_NAME);
    if (err < 0)
    {
        printk(KERN_CRIT "Failed to allocate character device region\n");
        return err;
    }

    // Create character device class
    dev_class = class_create(THIS_MODULE, DRIVER_NAME);
    if (IS_ERR(dev_class))
    {
        unregister_chrdev_region(dev_first, num_ctrls);
        printk(KERN_CRIT "Failed to create character device class\n");
        return PTR_ERR(dev_class);
    }

    // Register as PCI driver
    err = pci_register_driver(&driver);
    if (err != 0)
    {
        class_destroy(dev_class);
        unregister_chrdev_region(dev_first, num_ctrls);
        printk(KERN_ERR "Failed to register as PCI driver\n");
        return err;
    }

    printk(KERN_DEBUG DRIVER_NAME " loaded\n");
    return 0;
}
module_init(libnvm_helper_entry);


static void __exit libnvm_helper_exit(void)
{
    unsigned long remaining = 0;

    remaining = clear_map_list(&device_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu GPU memory mappings were still in use on unload\n", remaining);
    }

    remaining = clear_map_list(&host_list);
    if (remaining != 0)
    {
        printk(KERN_NOTICE "%lu host memory mappings were still in use on unload\n", remaining);
    }

    pci_unregister_driver(&driver);
    class_destroy(dev_class);
    unregister_chrdev_region(dev_first, num_ctrls);

    printk(KERN_DEBUG DRIVER_NAME " unloaded\n");
}
module_exit(libnvm_helper_exit);
