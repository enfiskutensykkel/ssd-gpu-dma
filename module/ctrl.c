#include "ctrl.h"
#include "list.h"
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <asm/errno.h>



struct ctrl* ctrl_get(struct list* list, struct class* cls, struct pci_dev* pdev, int number)
{
    struct ctrl* ctrl = NULL;

    ctrl = kmalloc(sizeof(struct ctrl), GFP_KERNEL | GFP_NOWAIT);
    if (ctrl == NULL)
    {
        printk(KERN_CRIT "Failed to allocate controller reference\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&ctrl->list);

    ctrl->pdev = pdev;
    ctrl->number = number;
    ctrl->rdev = 0;
    ctrl->cls = cls;
    ctrl->chrdev = NULL;

    snprintf(ctrl->name, sizeof(ctrl->name), "%s%d", KBUILD_MODNAME, ctrl->number);
    ctrl->name[sizeof(ctrl->name) - 1] = '\0';

    list_insert(list, &ctrl->list);

    return ctrl;
}



void ctrl_put(struct ctrl* ctrl)
{
    if (ctrl != NULL)
    {
        list_remove(&ctrl->list);
        ctrl_chrdev_remove(ctrl);
        kfree(ctrl);
    }
}



struct ctrl* ctrl_find_by_pci_dev(const struct list* list, const struct pci_dev* pdev)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (ctrl->pdev == pdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}



struct ctrl* ctrl_find_by_inode(const struct list* list, const struct inode* inode)
{
    const struct list_node* element = list_next(&list->head);
    struct ctrl* ctrl;

    while (element != NULL)
    {
        ctrl = container_of(element, struct ctrl, list);

        if (&ctrl->cdev == inode->i_cdev)
        {
            return ctrl;
        }

        element = list_next(element);
    }

    return NULL;
}



int ctrl_chrdev_create(struct ctrl* ctrl, dev_t first, const struct file_operations* fops)
{
    int err;
    struct device* chrdev = NULL;

    if (ctrl->chrdev != NULL)
    {
        printk(KERN_WARNING "Character device is already created\n");
        return 0;
    }

    ctrl->rdev = MKDEV(MAJOR(first), MINOR(first) + ctrl->number);

    cdev_init(&ctrl->cdev, fops);
    err = cdev_add(&ctrl->cdev, ctrl->rdev, 1);
    if (err != 0)
    {
        printk(KERN_ERR "Failed to add cdev\n");
        return err;
    }

    chrdev = device_create(ctrl->cls, NULL, ctrl->rdev, NULL, ctrl->name);
    if (IS_ERR(chrdev))
    {
        cdev_del(&ctrl->cdev);
        printk(KERN_ERR "Failed to create character device\n");
        return PTR_ERR(chrdev);
    }

    ctrl->chrdev = chrdev;

    printk(KERN_INFO "Character device /dev/%s created (%d.%d)\n",
            ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));

    return 0;
}



void ctrl_chrdev_remove(struct ctrl* ctrl)
{
    if (ctrl->chrdev != NULL)
    {
        device_destroy(ctrl->cls, ctrl->rdev);
        cdev_del(&ctrl->cdev);
        ctrl->chrdev = NULL;

        printk(KERN_DEBUG "Character device /dev/%s removed (%d.%d)\n",
                ctrl->name, MAJOR(ctrl->rdev), MINOR(ctrl->rdev));
    }
}

