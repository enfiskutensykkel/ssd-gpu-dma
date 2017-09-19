#include "ctrl_ref.h"
#include <linux/types.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <asm/atomic.h>
#include <asm/errno.h>
#ifdef _CUDA
#include <nv-p2p.h>
#endif


struct ctrl_ref* ctrl_ref_get(struct ctrl_ref* ref, struct ctrl_dev* dev)
{
    if (test_and_set_bit(1, &ref->in_use) == 0)
    {
        ref->ctrl = dev;
        ref->owner = current;
        ref->maps = NULL;
        return ref;
    }

    return NULL;
}


void ctrl_ref_put(struct ctrl_ref* ref)
{
    while (ref->maps != NULL)
    {
        //unmap_pages(ref->maps);
    }

    ref->owner = NULL;
    ref->maps = NULL;
    ref->ctrl = NULL;

    test_and_clear_bit(1, &ref->in_use);
}
