#ifndef __LIBNVM_HELPER_MAP_H__
#define __LIBNVM_HELPER_MAP_H__

#include "list.h"
#include <linux/types.h>
#include <linux/mm_types.h>


/* Forward declaration */
struct ctrl;
struct map;


typedef void (*release)(struct map*);


/*
 * Describes a range of mapped memory.
 */
struct map
{
    struct list_node    list;           /* Linked list header */
    struct task_struct* owner;          /* Owner of mapping */
    u64                 vaddr;          /* Starting virtual address */
    struct pci_dev*     pdev;           /* Reference to physical PCI device */
    unsigned long       page_size;      /* Logical page size */
    void*               data;           /* Custom data */
    release             release;        /* Custom callback for unmapping and releasing memory */
    unsigned long       n_addrs;        /* Number of mapped pages */
    uint64_t            addrs[1];       /* Bus addresses */
};



/*
 * Lock and map userspace pages for DMA.
 */
struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);



/*
 * Unmap and release memory.
 */
void unmap_and_release(struct map* map);



#ifdef _CUDA
/*
 * Lock and map GPU device memory.
 */
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages);
#endif



/*
 * Find memory mapping from vaddr and current task
 */
struct map* map_find(const struct list* list, u64 vaddr);


#endif /* __LIBNVM_HELPER_MAP_H__ */
