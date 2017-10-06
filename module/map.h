#ifndef __DIS_NVM_MODULE_MAP_H__
#define __DIS_NVM_MODULE_MAP_H__

#include "list.h"
#include <linux/types.h>
#include <linux/mm_types.h>

struct device;
struct ctrl_ref;


/*
 * Describes a range of mapped memory.
 *
 * Used to represent both mapped RAM memory and mapped GPU memory.
 */
struct map_descriptor
{
    struct map_list_head        list_head;      /* Linked list header */
    struct pci_dev*             pdev;           /* PCI device mappings are mapped for */
    unsigned long               page_size;      /* Virtual/logical page size */
    unsigned long               n_pages;        /* Number of pages pinned (should equal n_addrs) */
    void*                       pages;          /* Reference to the pinned pages */
    void*                       mappings;       /* Reference to mappings */
    unsigned long               n_addrs;        /* Number of bus addresses */
    dma_addr_t                  addrs[1];       /* Mapped bus addresses */
};


/*
 * Map user pages.
 *
 * Map n_pages of memory starting at virtual address vaddr for the controller.
 */
long map_user_pages(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map);


void unmap_user_pages(struct map_descriptor* map);


#ifdef _CUDA
long map_gpu_memory(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map);


void unmap_gpu_memory(struct map_descriptor* map);
#endif


/*
 * Look up a map from the list of user page mappings.
 */
struct map_descriptor* find_user_page_map(struct ctrl_ref* ref, u64 vaddr);


/*
 * Look up a map from the list of GPU mappings.
 */
struct map_descriptor* find_gpu_map(struct ctrl_ref* ref, u64 vaddr);


#endif /* __DIS_NVM_MODULE_MAP_H__ */
