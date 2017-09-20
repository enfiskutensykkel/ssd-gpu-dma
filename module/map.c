#include "map.h"
#include "list.h"
#include "ctrl_ref.h"
#include "ctrl_dev.h"
#include <linux/types.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/pci.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <asm/atomic.h>
#include <asm/errno.h>
#include <linux/dma-mapping.h>
#include <linux/version.h>
#include <linux/err.h>

#ifdef _CUDA
#include <nv-p2p.h>
#endif

#define IN_USE                  1


struct ctrl_ref* ctrl_ref_get(struct ctrl_ref* ref, struct ctrl_dev* dev)
{
    if (test_and_set_bit(IN_USE, &ref->flags) == 1)
    {
        // This slot is already taken
        return NULL;
    }

    ref->ctrl = dev;
    ref->owner = current;
    list_init(&ref->user_page_maps, 0);
    list_init(&ref->gpu_page_maps, 0);

    return ref;
}


void ctrl_ref_put(struct ctrl_ref* ref)
{
    struct map_descriptor* map;

    while (ref->gpu_page_maps.next != NULL)
    {
    }

    while (ref->user_page_maps.next != NULL)
    {
        map = (struct map_descriptor*) ((void*) ref->user_page_maps.next);
        unmap_user_pages(ref, map);
    }

    ref->owner = NULL;
    ref->ctrl = NULL;

    test_and_clear_bit(IN_USE, &ref->flags);
}


static long lock_user_pages(struct map_descriptor* map, struct task_struct* task, u64 vaddr, unsigned long n_pages)
{
    long retval;
    struct page** pages;

    // Allocate page array
    pages = (struct page**) kcalloc(n_pages, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_ERR "Failed to allocate user page array\n");
        return -ENOMEM;
    }

    // Pin pages in memory
#if (LINUX_VERSION_CODE <= KERNEL_VERSION(4, 9, 0))
    retval = get_user_pages(task, task->mm, vaddr, n_pages, 1, 0, pages, NULL);
#else
    retval = get_user_pages(task, task->mm, vaddr_start, n_pages, FOLL_WRITE, pages, NULL);
#endif

    if (retval <= 0)
    {
        kfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    map->n_pages = retval;
    map->pages = (void*) pages;

    printk(KERN_DEBUG "Locked %lu pages in memory starting at virtual address %llx\n",
            map->n_pages, vaddr);

    return 0;
}


long map_user_pages(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map)
{
    unsigned long i;
    long err;
    struct map_descriptor* md;
    struct page** pages;
    struct device* dev;

    if (map != NULL)
    {
        *map = NULL;
    }

    if (n_pages < 1)
    {
        return -EINVAL;
    }

    // Allocate map descriptor
    md = kmalloc(sizeof(struct map_descriptor) + (n_pages - 1) * sizeof(dma_addr_t), GFP_KERNEL);
    if (md == NULL)
    {
        printk(KERN_ERR "Failed to allocate map descriptor\n");
        return -ENOMEM;
    }

    md->page_size = PAGE_SIZE;

    // Initialize list header
    vaddr &= PAGE_MASK;
    list_init(md, vaddr);
    
    // Pin pages to memory
    err = lock_user_pages(md, ref->owner, vaddr, n_pages);
    if (err != 0)
    {
        kfree(md);
        return err;
    }

    // Traverse pages and map them for controller
    dev = &ref->ctrl->pdev->dev;
    pages = (struct page**) md->pages;
    md->n_addrs = md->n_pages;
    for (i = 0; i < md->n_pages; ++i)
    {
        md->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    // Insert into linked list
    list_insert(&ref->user_page_maps, md);

    if (map != NULL)
    {
        *map = md;
    }

    printk(KERN_INFO "Mapped %lu host pages for controller\n", md->n_addrs);
    return 0;
}


void unmap_user_pages(struct ctrl_ref* ref, struct map_descriptor* map)
{
    unsigned long i;
    struct device* dev;
    struct page** pages;

    // Remove from linked list
    list_remove(map);

    // Unmap pages for controller
    dev = &ref->ctrl->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    printk(KERN_DEBUG "Unmapped %lu host pages for controller\n", i);

    // Unpin pages
    pages = (struct page**) map->pages;
    for (i = 0; i < map->n_pages; ++i)
    {
        put_page(pages[i]);
    }

    kfree(map->pages);
    kfree(map);

    printk(KERN_DEBUG "Released %lu pages\n", i);
}

