#ifndef __DIS_NVM_MODULE_LIST_H__
#define __DIS_NVM_MODULE_LIST_H__

#include <linux/types.h>


/*
 * Put this in the top of a struct.
 *
 * Assumes that the head of the list is present and that the head is 
 * unremovable.
 */
struct map_list_head
{
    struct map_list_head*           next;           /* Pointer to next element in list */
    struct map_list_head*           prev;           /* Pointer to previous element in list */
    u64                             vaddr;          /* Sort list by virtual address ranges */
};


/*
 * Initialize the list portion of the node.
 */
void list_init(void* node, u64 vaddr);


/*
 * Insert the node in to an existing list.
 */
long list_insert(struct map_list_head* list_start, void* node);


/*
 * Find an element in the list and return it.
 */
void* list_find(struct map_list_head* list_start, u64 vaddr);


/*
 * Remove an element from the list.
 */
void list_remove(void* node);


#endif /* __DIS_NVM_MODULE_LIST_H__ */
