#include "ctrl_dev.h"
#include <linux/types.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/pci.h>
#include <linux/device.h>


int ctrl_dev_create(struct ctrl_dev* dev, 
                    int num, 
                    dev_t first, 
                    struct class* cls, 
                    const struct file_operations* fops, 
                    const char* name)
{
    int err;
    struct device* df;

    dev_t devt = MKDEV(MAJOR(first), MINOR(first) + num);

    snprintf(dev->name, sizeof(dev->name), "%s%d", name, num);
    dev->name[sizeof(dev->name) - 1] = '\0';

    cdev_init(&dev->cdev, fops);
    err = cdev_add(&dev->cdev, devt, 1);
    if (err != 0)
    {
        printk(KERN_ERR "Failed to add cdev\n");
        return err;
    }

    df = device_create(cls, NULL, devt, NULL, dev->name);
    if (IS_ERR(df))
    {
        cdev_del(&dev->cdev);
        printk(KERN_ERR "Failed to create chrdev\n");
        return PTR_ERR(df);
    }

    dev->devt = devt;
    dev->cls = cls;

    printk(KERN_DEBUG "chrdev %s created (%d.%d)\n", 
            dev->name, MAJOR(devt), MINOR(devt));
    return 0;
}


void ctrl_dev_remove(struct ctrl_dev* dev)
{
    device_destroy(dev->cls, dev->devt);
    cdev_del(&dev->cdev);

    printk(KERN_DEBUG "chrdev %s removed (%d.%d)\n",
            dev->name, MAJOR(dev->devt), MINOR(dev->devt));
}
