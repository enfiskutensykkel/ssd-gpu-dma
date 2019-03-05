#ifndef __NVM_INTERNAL_DIS_MAP_H__
#define __NVM_INTERNAL_DIS_MAP_H__
#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <stdbool.h>
#include "dma.h"
#include <sisci_types.h>


/*
 * What kind of memory are we mapping.
 */
enum segment_type
{
    SEGMENT_TYPE_LOCAL      =   0x1,    // Local segment
    SEGMENT_TYPE_REMOTE     =   0x2,    // Remote segment
    SEGMENT_TYPE_DEVICE     =   0x3,    // Device segment (requires connect/disconnect)
    SEGMENT_TYPE_PHYSICAL   =   0x4     // Physical memory segment
};



/*
 * Describe mapping for device.
 */
struct map
{
    enum segment_type       type;       // Kind of segment
    sci_local_segment_t     lseg;       // Local segment descriptor
    sci_remote_segment_t    rseg;       // Remote segment descriptor
    uint32_t                adapter;    // DIS adapter number
    size_t                  size;       // Segment size
    bool                    mapped;     // Is segment mapped into virtual address space
    sci_map_t               md;         // SISCI mapping descriptor
    struct va_range         range;      // Memory range descriptor
};


#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_MAP_H__ */
