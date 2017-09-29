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

#define GPU_PAGE_SHIFT          16
#define GPU_PAGE_SIZE           (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK           ~(GPU_PAGE_SIZE - 1)
#define IN_USE_BIT              1


struct ctrl_ref* ctrl_ref_get(struct ctrl_ref* ref, struct ctrl_dev* dev)
{
    if (test_and_set_bit(IN_USE_BIT, &ref->flags) == 1)
    {
        // This slot is already taken
        return NULL;
    }

    ref->ctrl = dev;
    ref->owner = current->pid;
    list_init(&ref->user_page_maps, 0);
    list_init(&ref->gpu_page_maps, 0);

    return ref;
}


void ctrl_ref_put(struct ctrl_ref* ref)
{
    struct map_descriptor* map;

#ifdef _CUDA
    while (ref->gpu_page_maps.next != NULL)
    {
        map = (struct map_descriptor*) ((void*) ref->gpu_page_maps.next);
        unmap_gpu_memory(map);
    }
#endif

    while (ref->user_page_maps.next != NULL)
    {
        map = (struct map_descriptor*) ((void*) ref->user_page_maps.next);
        unmap_user_pages(map);
    }

    ref->owner = 0;
    ref->ctrl = NULL;

    test_and_clear_bit(IN_USE_BIT, &ref->flags);
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

    return 0;
}


#ifdef _CUDA
static void free_callback(struct map_descriptor* map)
{
    nvidia_p2p_free_page_table((nvidia_p2p_page_table_t*) map->pages);
    map->pages = NULL;
    printk(KERN_NOTICE "Force released P2P page table\n");
}
#endif


#ifdef _CUDA
long map_gpu_memory(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map)
{
    unsigned long i;
    long err;
    struct map_descriptor* md;
    nvidia_p2p_page_table_t* pages;

    *map = NULL;

    if (n_pages < 1)
    {
        return -EINVAL;
    }

    md = kmalloc(sizeof(struct map_descriptor) + (n_pages - 1) * sizeof(dma_addr_t), GFP_KERNEL);
    if (md == NULL)
    {
        printk(KERN_ERR "Failed to allocate map descriptor\n");
        return -ENOMEM;
    }

    vaddr &= GPU_PAGE_MASK;
    md->page_size = GPU_PAGE_SIZE;
    md->dev = &ref->ctrl->pdev->dev;

    list_init(md, vaddr);

    err = nvidia_p2p_get_pages(0, 0, vaddr, GPU_PAGE_SIZE * n_pages, 
            (nvidia_p2p_page_table_t**) &md->pages, (void (*)(void*)) free_callback, md);

    if (err != 0)
    {
        kfree(md);
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %ld\n", err);
        return err;
    }

    pages = (nvidia_p2p_page_table_t*) md->pages;
    md->n_pages = pages->entries;
    md->n_addrs = 0;

    for (i = 0; i < md->n_pages; ++i)
    {
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(4, 9, 0))
#warning "dma_map_resource() is not tested"
        md->addrs[i] = dma_map_resource(md->dev, pages->pages[i]->physical_address, GPU_PAGE_SIZE, DMA_BIDIRECTIONAL, 0);
        
        err = dma_mapping_errors(dev, md->addrs[i]);
        if (err != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            break;
        }
#else
        md->addrs[i] = pages->pages[i]->physical_address;
#endif

        md->n_addrs++;
    }

    list_insert(&ref->gpu_page_maps, md);

    *map = md;
    printk(KERN_INFO "Mapped %lu GPU pages (pid %d)\n", md->n_addrs, current->pid);
    return 0;
}
#endif


#ifdef _CUDA
void unmap_gpu_memory(struct map_descriptor* map)
{
    unsigned long i;

    list_remove(map);

#if (LINUX_VERSION_CODE >= KERNEL_VERSION(4, 9, 0))
#warning "dma_unmap_resource() is not tested"
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_resource(map->dev, map->addrs[i], GPU_PAGE_SIZE, DMA_BIDIRECTIONAL, 0);
    }

#endif

    i = map->n_pages;

    if (map->pages != NULL)
    {
        printk(KERN_DEBUG "GPU pages were already removed\n");
        nvidia_p2p_put_pages(0, 0, map->list_head.vaddr, (nvidia_p2p_page_table_t*) map->pages);
    }
    kfree(map);

    printk(KERN_DEBUG "Unmapped %lu GPU pages (pid %d)\n", i, current->pid);
}
#endif


long map_user_pages(struct ctrl_ref* ref, u64 vaddr, unsigned long n_pages, struct map_descriptor** map)
{
    unsigned long i;
    long err;
    struct map_descriptor* md;
    struct page** pages;

    *map = NULL;

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

    vaddr &= PAGE_MASK;
    md->page_size = PAGE_SIZE;

    // Initialize list header
    list_init(md, vaddr);
    
    // Pin pages to memory
    err = lock_user_pages(md, /*ref->owner*/ current, vaddr, n_pages);
    if (err != 0)
    {
        kfree(md);
        return err;
    }

    // Traverse pages and map them for controller
    md->dev = &ref->ctrl->pdev->dev;
    pages = (struct page**) md->pages;
    for (i = 0; i < md->n_pages; ++i)
    {
        md->addrs[i] = dma_map_page(md->dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        err = dma_mapping_error(md->dev, md->addrs[i]);
        if (err != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            break;
        }
    }
    md->n_addrs = i;

    // Insert into linked list
    list_insert(&ref->user_page_maps, md);

    *map = md;
    printk(KERN_INFO "Mapped %lu host pages (pid %d)\n", md->n_addrs, current->pid);
    return 0;
}


void unmap_user_pages(struct map_descriptor* map)
{
    unsigned long i;
    struct page** pages;

    // Remove from linked list
    list_remove(map);

    // Unmap pages for controller
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(map->dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    // Unpin pages
    pages = (struct page**) map->pages;
    for (i = 0; i < map->n_pages; ++i)
    {
        put_page(pages[i]);
    }

    kfree(map->pages);
    kfree(map);

    printk(KERN_DEBUG "Unmapped %lu host pages (pid %d)\n", i, current->pid);
}


struct map_descriptor* find_user_page_map(struct ctrl_ref* ref, u64 vaddr)
{
    vaddr &= PAGE_MASK;
    return (struct map_descriptor*) list_find(&ref->user_page_maps, vaddr);
}


struct map_descriptor* find_gpu_map(struct ctrl_ref* ref, u64 vaddr)
{
    vaddr &= GPU_PAGE_MASK;
    return (struct map_descriptor*) list_find(&ref->gpu_page_maps, vaddr);
}

