#ifndef __DIS_NVM_MODULE_CTRL_REF_H__
#define __DIS_NVM_MODULE_CTRL_REF_H__

#include "list.h"
#include <linux/types.h>
#include <linux/sched.h>
#include <linux/fs.h>

struct ctrl_dev;


/*
 * NVMe controller reference.
 *
 * A userspace process will typically acquire a reference to the actual
 * NVMe controller, in order to read and write to registers.
 */
struct ctrl_ref
{
    unsigned long               flags;          /* 0=slot is free, 1=slot is taken */
    struct ctrl_dev*            ctrl;           /* Controller device */
    pid_t                       owner;          /* Userspace process that owns this reference */
    struct map_list_head        user_page_maps; /* Linked list of user pages */
    struct map_list_head        gpu_page_maps;  /* Linked list of mapped GPU memory */
};



/*
 * Acquire a controller reference.
 */
struct ctrl_ref* ctrl_ref_get(struct ctrl_ref* ref, struct ctrl_dev* dev);


/*
 * Release a controller reference.
 */
void ctrl_ref_put(struct ctrl_ref* ref);


#endif /* __DIS_NVM_MODULE_CTRL_REF_H__ */
