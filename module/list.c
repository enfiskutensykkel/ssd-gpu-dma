#include "list.h"
#include <linux/types.h>
#include <linux/printk.h>
#include <asm/errno.h>


void list_init(void* node, u64 vaddr)
{
    struct map_list_head* ptr = (struct map_list_head*) node;

    ptr->next = NULL;
    ptr->prev = NULL;
    ptr->vaddr = vaddr;
}


void* list_find(struct map_list_head* list_start, u64 vaddr)
{
    struct map_list_head* ptr;
   
    // Start on the first element after head
    ptr  = list_start->next;

    // Loop through list until we encounter addresses larger than the
    // one we are searching for
    while (ptr != NULL && ptr->vaddr <= vaddr)
    {
        if (ptr->vaddr == vaddr)
        {
            return ptr;
        }

        ptr = ptr->next;
    }

    return NULL;
}


void list_remove(void* node)

{
    struct map_list_head* curr = (struct map_list_head*) node;
    struct map_list_head* next = curr->next;
    struct map_list_head* prev = curr->prev;

    if (prev != NULL)
    {
        prev->next = next;
    }

    if (next != NULL)
    {
        next->prev = prev;
    }

    curr->next = NULL;
    curr->prev = NULL;
}


long list_insert(struct map_list_head* list_start, void* node)
{
    long count = 0;
    struct map_list_head* insert = (struct map_list_head*) node;
    struct map_list_head* curr = list_start;
    struct map_list_head* next = curr->next;

    // Iterate until we find a suitable spot
    while (next != NULL && next->vaddr < insert->vaddr)
    {
        curr = next;
        next = next->next;
        ++count;
    }

    // Some sanity checking
    if (curr->vaddr == insert->vaddr)
    {
        printk(KERN_WARNING "Virtual address is already present in list at position %ld: %llx\n", 
                count, insert->vaddr);
        return -EEXIST;
    }

    // Insert in list
    curr->next = insert;
    insert->prev = curr;
    insert->next = next;
    if (next != NULL)
    {
        next->prev = insert;
    }

    // Return position of node
    return count;
}

