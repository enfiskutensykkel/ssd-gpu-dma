#ifndef __NVM_MODULE_CTRL_DEV_H__
#define __NVM_MODULE_CTRL_DEV_H__

#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>

struct ctrl_dev
{
    unsigned long               in_use;         /* Indicates if this struct is used */
    const struct pci_dev*       pdev;           /* Reference to physical PCI device */
    char                        name[64];       /* Device name */
    dev_t                       rdev;           /* Device register */
    struct class*               cls;            /* Device class */
    struct cdev                 cdev;           /* Character device */
    struct device*              chrdev;         /* Character device handle */
};


/* 
 * Initialize controller device handle.
 */
void ctrl_dev_reset(struct ctrl_dev* dev, 
                    dev_t first, 
                    int num, 
                    struct class* cls,
                    const char* name_prefix);


/*
 * Acquire a controller device
 */
struct ctrl_dev* ctrl_dev_get(struct ctrl_dev* dev, const struct pci_dev* pdev);


/*
 * Release controller device
 */
void ctrl_dev_put(struct ctrl_dev* dev);


/*
 * Create character device and set up file operations.
 */
int ctrl_dev_chrdev_create(struct ctrl_dev* dev,
                           const struct file_operations* fops);


/*
 * Remove character device.
 */
void ctrl_dev_chrdev_remove(struct ctrl_dev* dev);


#endif
