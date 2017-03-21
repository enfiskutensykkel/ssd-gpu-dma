#ifndef __MEMORY_H__
#define __MEMORY_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>


/* 
 * Pinned memory handle
 *
 * Descriptor to a pinned memory range.
 */
struct memory
{
    long            kernel_handle;  // Handle retrieved from kernel module
    int             device;         // CUDA device (-1 if no device)
    void*           virt_addr;      // Pointer to memory
    size_t          range_size;     // Total memory range size
    size_t          page_size;      // Size of a page unit
    size_t          n_addrs;        // Number of bus addresses
    uint64_t        bus_addr[0];    // Bus addresses to pinned memory
};


/* Convenience type */
typedef struct memory memory_t;


/*
 * Page handle
 *
 * Descriptor to a page-locked page
 */
struct page
{
    long            kernel_handle;  // Handle retrieved from kernel module
    int             device;         // CUDA device (-1 if no device)
    void*           virt_addr;      // Pointer to memory
    size_t          page_size;      // Size of a page unit
    uint64_t        bus_addr;       // Bus address to page
};


/* Convenience type */
typedef struct page page_t;


#ifdef __cplusplus
}
#endif
#endif
