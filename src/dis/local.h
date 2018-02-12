#ifndef __NVM_INTERNAL_DIS_LOCAL_MEMORY_H__
#define __NVM_INTERNAL_DIS_LOCAL_MEMORY_H__

/* Forward declarations */
struct local_memory;


#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <stddef.h>
#include <stdint.h>
#include <sisci_types.h>



/*
 * Local memory descriptor.
 *
 * Describes a local memory segment and its mapping into address space.
 */
struct local_memory
{
    sci_desc_t              sd;             // SISCI virtual device descriptor
    sci_local_segment_t     segment;        // SISCI local segment descriptor
    uint32_t                id;             // Local segment identifier
    size_t                  size;           // Size of local memory
    uint32_t                flags;          // SISCI flags used when creating segment
};



/*
 * Create local segment and map it into local address space.
 */
int _nvm_local_memory_get(struct local_memory* mem, uint32_t id, size_t size);



/*
 * Create local segment with attached memory.
 * This function will NOT map segment into address space.
 */
int _nvm_local_memory_get_attached(struct local_memory* mem, uint32_t id, void* ptr, size_t size);



/*
 * Create local segment with registered host memory.
 * This function will not map segment into address space (as it is already mapped).
 */
int _nvm_local_memory_get_registered(struct local_memory* mem, uint32_t id, void* ptr, size_t size);


/*
 * Remove local segment.
 */
void _nvm_local_memory_put(struct local_memory* mem);



#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_LOCAL_MEMORY_H__ */
