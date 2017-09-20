#ifndef __DIS_NVM_MODULE_MAP_H__
#define __DIS_NVM_MODULE_MAP_H__

#include "list.h"
#include <linux/types.h>
#include <linux/mm_types.h>

struct ctrl_ref;


/*
 * Describes a range of mapped memory.
 *
 * Used to represent both mapped RAM memory and mapped GPU memory.
 */
struct map_descriptor
{
    struct list_head            list_head;      /* Linked list header */
    unsigned long               page_size;      /* Virtual/logical page size */
    unsigned long               n_pages;        /* Number of pages pinned (should equal n_addrs) */
    void*                       pages;          /* Reference to the pinned pages */
    unsigned long               n_addrs;        /* Number of bus addresses */
    dma_addr_t                  addrs[1];       /* Mapped bus addresses */
};


/*
 * Map user pages.
 *
 * Map n_pages of memory starting at virtual address vaddr for the controller.
 */
long map_user_pages(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map);


/*
 * 
 */
void unmap_user_pages(struct ctrl_ref* ref, struct map_descriptor* map);


/*
 * Look up a map from the list of user page mappings.
 */
struct map_descriptor* find_user_page_map(struct ctrl_ref* ref, u64 vaddr);


/*
 * Look up a map from the list of GPU mappings.
 */
//struct map_descriptor* find_gpu_page_map(struct ctrl_ref* ref, u64 vaddr);




//long map_device_memory(struct ctrl_ref* ref, u64 addr, size_t n_pages, u64 __user* ioaddrs);



#endif /* __DIS_NVM_MODULE_MAP_H__ */
