#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>

#ifdef _CUDA
#include <nv-p2p.h>

struct gpu_region
{
    nvidia_p2p_page_table_t* pages;
    nvidia_p2p_dma_mapping_t* mappings;
};
#endif


#define GPU_PAGE_SHIFT  16
#define GPU_PAGE_SIZE   (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK   ~(GPU_PAGE_SIZE - 1)



static struct map* create_descriptor(const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map* map = NULL;

    map = kmalloc(sizeof(struct map) + (n_pages - 1) * sizeof(uint64_t), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);

    map->owner = current;
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;

    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}



void unmap_and_release(struct map* map)
{
    list_remove(&map->list);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
    }

    kfree(map);
}



struct map* map_find(const struct list* list, u64 vaddr)
{
    const struct list_node* element = list_next(&list->head);
    struct map* map = NULL;

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner == current)
        {
            if (map->vaddr == (vaddr & (~(map->page_size - 1))))
            {
                return map;
            }
        }

        element = list_next(element);
    }

    return NULL;
}



static void release_user_pages(struct map* map)
{
    unsigned long i;
    struct page** pages;
    struct device* dev;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page**) map->data;
    for (i = 0; i < map->n_addrs; ++i)
    {
        put_page(pages[i]);
    }

    kfree(map->data);
    map->data = NULL;

    //printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
}



static long map_user_pages(struct map* map)
{
    unsigned long i;
    long retval;
    struct page** pages;
    struct device* dev;

    pages = (struct page**) kcalloc(map->n_addrs, sizeof(struct page*), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 5, 7)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(current, current->mm, map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4, 8, 17)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#else
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);
#endif
    if (retval <= 0)
    {
        kfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    if (map->n_addrs != retval)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %ld\n", map->n_addrs, retval);
    }
    map->n_addrs = retval;
    map->page_size = PAGE_SIZE;
    map->data = (void*) pages;
    map->release = release_user_pages;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        retval = dma_mapping_error(dev, map->addrs[i]);
        if (retval != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            return retval;
        }
    }

    return 0;
}



struct map* map_userspace(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map* md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = PAGE_SIZE;

    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}



#ifdef _CUDA
static void force_release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            nvidia_p2p_free_dma_mapping(gd->mappings);
        }

        if (gd->pages != NULL)
        {
            nvidia_p2p_free_page_table(gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}
#endif



#ifdef _CUDA
void release_gpu_memory(struct map* map)
{
    struct gpu_region* gd = (struct gpu_region*) map->data;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            nvidia_p2p_dma_unmap_pages(map->pdev, gd->pages, gd->mappings);
        }

        if (gd->pages != NULL)
        {
            nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        //printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}
#endif



#ifdef _CUDA
int map_gpu_memory(struct map* map)
{
    unsigned long i;
    int err;
    struct gpu_region* gd;

    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->pages = NULL;
    gd->mappings = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_memory;

    err = nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages, 
            (void (*)(void*)) force_release_gpu_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "nvidia_p2p_get_pages() failed: %d\n", err);
        return err;
    }

    err = nvidia_p2p_dma_map_pages(map->pdev, gd->pages, &gd->mappings);
    if (err != 0)
    {
        printk(KERN_ERR "nvidia_p2p_dma_map_pages() failed: %d\n", err);
        return err;
    }

    if (map->n_addrs != gd->pages->entries)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, gd->pages->entries);
    }

    map->n_addrs = gd->pages->entries;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = gd->mappings->dma_addresses[i];
    }
    
    return 0;
}
#endif



#ifdef _CUDA
struct map* map_device_memory(struct list* list, const struct ctrl* ctrl, u64 vaddr, unsigned long n_pages)
{
    int err;
    struct map* md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;

    err = map_gpu_memory(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    //printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n", 
    //        md->n_addrs, md->vaddr);
    return md;
}
#endif

