#ifndef __LIBNVM_HELPER_CTRL_H__
#define __LIBNVM_HELPER_CTRL_H__

#include "list.h"
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/device.h>


/*
 * Represents an NVM controller.
 */
struct ctrl
{
    struct list_node    list;       /* Linked list head */
    struct pci_dev*     pdev;       /* Reference to physical PCI device */
    char                name[64];   /* Character device name */
    int                 number;     /* Controller number */
    dev_t               rdev;       /* Character device register */
    struct class*       cls;        /* Character device class */
    struct cdev         cdev;       /* Character device */
    struct device*      chrdev;     /* Character device handle */
};



/*
 * Acquire a controller reference.
 */
struct ctrl* ctrl_get(struct list* list, struct class* cls, struct pci_dev* pdev, int number);



/*
 * Release controller reference.
 */
void ctrl_put(struct ctrl* ctrl);



/*
 * Find controller device.
 */
struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev);



/*
 * Find controller reference.
 */
struct ctrl* ctrl_find_by_inode(const struct list* list, const struct inode* inode);



/*
 * Create character device and set up file operations.
 */
int ctrl_chrdev_create(struct ctrl* ctrl, 
                       dev_t first,
                       const struct file_operations* fops);



/*
 * Remove character device.
 */
void ctrl_chrdev_remove(struct ctrl* ctrl);



#endif /* __LIBNVM_HELPER_CTRL_H__ */
