#ifndef __NVM_MODULE_CTRL_DEV_H__
#define __NVM_MODULE_CTRL_DEV_H__

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>

struct ctrl_dev
{
    const struct pci_dev*       ctrl_pdev;      /* Reference to physical PCI device */
    char                        name[64];       /* Device name */
    dev_t                       devt;           /* Device register */
    struct class*               cls;            /* Device class */
    struct cdev                 cdev;           /* Character device */
    struct device*              dev;            /* Device handle */
};


/*
 * Create character device and set up file operations.
 */
int ctrl_dev_create(struct ctrl_dev* dev, 
                    int num, 
                    dev_t first, 
                    struct class* cls, 
                    const struct file_operations* fops, 
                    const char* name
                    );


/*
 * Remove character device.
 */
void ctrl_dev_remove(struct ctrl_dev* dev);


#endif
