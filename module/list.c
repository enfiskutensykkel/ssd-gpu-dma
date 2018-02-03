#include "list.h"
#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/printk.h>
#include <asm/errno.h>



void list_init(struct list* list)
{
    list_node_init(&list->head);
    list->head.prev = &list->head;

    spin_lock_init(&list->lock);
}



void list_remove(struct list_node* element)
{
    if (element != NULL && element->list != NULL)
    {
        spin_lock(&element->list->lock);

        if (element->prev != NULL)
        {
            element->prev->next = element->next;
        }

        if (element->next != NULL)
        {
            element->next->prev = element->prev;
        }

        spin_unlock(&element->list->lock);

        element->list = NULL;
        element->next = NULL;
        element->prev = NULL;
    }
}



void list_insert(struct list* list, struct list_node* element)
{
    struct list_node* last = NULL;

    spin_lock(&list->lock);
    last = list->head.prev;
    last->next = element;

    element->list = list;
    element->prev = last;
    element->next = NULL;

    list->head.prev = element;

    spin_unlock(&list->lock);
}

