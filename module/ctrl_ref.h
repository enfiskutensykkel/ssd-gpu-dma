#ifndef __NVM_MODULE_CTRL_REF_H__
#define __NVM_MODULE_CTRL_REF_H__

#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/mm_types.h>
#include <linux/sched.h>
#include <linux/pci.h>
#ifdef _CUDA
#include <nv-p2p.h>
#endif

struct mem_map;


/*
 * NVMe controller reference.
 *
 * A userspace process will typically acquire a reference to the actual
 * NVMe controller, in order to read and write to registers.
 */
struct ctrl_ref
{
    struct task_struct*         holder;         /* Userspace process that holds this reference */
    struct mem_map*             maps;           /* Linked list of memory mappings */
};



/*
 * Memory map reference.
 *
 * Holds information about a memory mapping. A user process passes a pointer
 * to virtual memory and we need to look up the bus addresses and set up IOMMU
 * mappings.
 */
struct mem_map
{
    spinlock_t                  lock;           /* Lock structure when making changes */
    struct mem_map*             next;           /* Pointer to the next memory map reference */
    struct mem_map*             prev;           /* Pointer to the previous memory map reference */
    u64                         vaddr_start;    /* Start address of virtual memory, used to look up the mapping again */
    long                        n_pages;        /* Number of pages (range) */
    struct page**               host_pages;     /* User pages */
    struct vm_area_struct**     vm_areas;       /* Virtual memory mappings descriptor */
#ifdef _CUDA
    nvidia_p2p_page_table_t*    dev_pages;      /* GPU pages */
#endif
};



/*
 * Initialize a controller reference.
 */
int ctrl_ref_init(struct ctrl_ref** ctrl, const struct pci_dev* pdev, struct vm_area_struct* vma);


/*
 * Release a controller reference.
 */
void ctrl_ref_free(struct ctrl_ref** ctrl);


/*
 * Map host pages for controller.
 * Returns handle.
 */
long map_host_pages(struct ctrl_ref* ctrl, u64 vaddr, long n_pages, struct mem_map** map);


/* 
 * Map GPU pages for controller.
 * Returns handle.
 */
long map_dev_pages(struct ctrl_ref* ctrl, u64 vaddr, long n_pages, struct mem_map** map);


/*
 * Look up a memory mapping.
 */
struct mem_map* find_map(struct ctrl_ref* ctrl, u64 vaddr);


/*
 * Unmap pages.
 */
int unmap_pages(struct mem_map* map);


#endif /* __NVM_MODULE_DEVREF_H__ */
