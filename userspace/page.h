#ifndef __PAGE_H__
#define __PAGE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>


/* Page handle
 *
 * Represent a page-locked page of memory.
 */
struct page
{
    int             device;         // CUDA device page is hosted on (-1 for no device)
    long            kernel_handle;  // handle retrieved from kernel module
    void*           virt_addr;      // virtual address of page
    uint64_t        phys_addr;      // physical address of page
    size_t          page_size;      // page size
};


/* Convenience type */
typedef struct page page_t;


/* 
 * Allocate a page of memory and pin it so it remains in memory.
 */
int get_page(page_t* page, int ioctl_fd, int device);


/*
 * Release an allocated page.
 */
void put_page(page_t* page, int ioctl_fd);


#ifdef __cplusplus
}
#endif
#endif
