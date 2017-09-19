#include "ctrl_dev.h"
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/atomic.h>
#include <asm/errno.h>


void ctrl_dev_reset(struct ctrl_dev* dev,
                    dev_t first,
                    int num,
                    struct class* cls,
                    const char* name)
{
    snprintf(dev->name, sizeof(dev->name), "%s%d", name, num);
    dev->name[sizeof(dev->name) - 1] = '\0';

    clear_bit(1, &dev->in_use);
    dev->pdev = NULL;
    dev->rdev = MKDEV(MAJOR(first), MINOR(first) + num);
    dev->cls = cls;
    dev->chrdev = NULL;
}


struct ctrl_dev* ctrl_dev_get(struct ctrl_dev* dev, const struct pci_dev* pdev)
{
    if (test_and_set_bit(1, &dev->in_use) == 0)
    {
        dev->pdev = pdev;
        return dev;
    }

    return NULL;
}


void ctrl_dev_put(struct ctrl_dev* dev)
{
    ctrl_dev_chrdev_remove(dev);
    dev->pdev = NULL;

    test_and_clear_bit(1, &dev->in_use);
}


int ctrl_dev_chrdev_create(struct ctrl_dev* dev, 
                    const struct file_operations* fops)
{
    int err;
    struct device* chrdev;

    if (dev->chrdev != NULL)
    {
        printk(KERN_WARNING "Character device is already created\n");
        return 0;
    }

    if (dev->pdev == NULL)
    {
        printk(KERN_ERR "Attempting to create character device from removed controller\n");
        return -EBADF;
    }

    cdev_init(&dev->cdev, fops);
    err = cdev_add(&dev->cdev, dev->rdev, 1);
    if (err != 0)
    {
        printk(KERN_ERR "Failed to add cdev\n");
        return err;
    }

    chrdev = device_create(dev->cls, NULL, dev->rdev, NULL, dev->name);
    if (IS_ERR(chrdev))
    {
        cdev_del(&dev->cdev);
        printk(KERN_ERR "Failed to create character device\n");
        return PTR_ERR(chrdev);
    }

    dev->chrdev = chrdev;

    printk(KERN_DEBUG "Character device /dev/%s created (%d.%d)\n", 
            dev->name, MAJOR(dev->rdev), MINOR(dev->rdev));

    return 0;
}


void ctrl_dev_chrdev_remove(struct ctrl_dev* dev)
{
    if (dev->chrdev != NULL)
    {
        device_destroy(dev->cls, dev->rdev);
        cdev_del(&dev->cdev);
        dev->chrdev = NULL;

        printk(KERN_DEBUG "Character device /dev/%s removed (%d.%d)\n",
                dev->name, MAJOR(dev->rdev), MINOR(dev->rdev));
    }
}

