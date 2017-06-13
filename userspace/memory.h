#ifndef __MEMORY_H__
#define __MEMORY_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <sisci_types.h>


/* 
 * Memory range handle
 *
 * Describes an arbitrarily large memory buffer. Contains the bus/physical 
 * addresses to the page-locked memory "units" used by the buffer.
 *
 * Note: A "unit" corresponds to NVM Memory Page Size (MPS)
 */
struct mem_buffer
{
    sci_desc_t          sd;
    sci_local_segment_t segment;
    sci_map_t           mapping;
    uint64_t            device_handle;  
    int                 id;             // buffer id
    int                 device;         // CUDA device (-1 if no device)
    void*               virt_addr;      // Pointer to virtual memory (may be device memory)
    size_t              range_size;     // Total memory range size
    size_t              page_size;      // Size of a physical page
    size_t              unit_size;      // Size of a logical memory unit
    uint64_t            bus_handle;
    size_t              n_addrs;        // Number of memory units
    uint64_t            bus_addr[0];    // Bus/physical addresses to memory units
};


/* Convenience type */
typedef struct mem_buffer buffer_t;


/*
 * Page handle
 *
 * Descriptor to a page of memory. Contains the bus/physical address
 * to the page-locked page.
 */
struct mem_page
{
    sci_desc_t          sd;
    sci_local_segment_t segment;
    sci_map_t           mapping;
    int                 id;             // buffer id
    int                 device;         // CUDA device (-1 if no device)
    void*               virt_addr;      // Pointer to memory (may be device memory)
    size_t              page_size;      // Size of a page unit
    uint64_t            bus_handle;
    uint64_t            bus_addr;       // Bus/physical address to page
};


/* Convenience type */
typedef struct mem_page page_t;



/* Allocate a memory buffer
 *
 * Allocate a memory range and pin it in memory.
 *
 * Note: The requested size will be rounded up to the nearest
 * page-aligned address.
 *
 * Returns a handle on success or NULL on failure.
 */
buffer_t* get_buffer(int device, int id, size_t buffer_size, size_t mem_unit_size, uint64_t bus_handle);


/* Release memory buffer
 *
 * Unpin memory and free allocated memory.
 * After calling this function the handle is invalid.
 */
void put_buffer(buffer_t* buffer_handle);


/* Allocate a page of memory and pin it
 *
 * Returns 0 on success and sets the page handle, or an errno
 * on failure.
 */
int get_page(int device, int id, page_t* page_handle, uint64_t bus_handle);


/* Release allocated page */
void put_page(page_t* page_handle);


#ifdef __cplusplus
}
#endif
#endif
