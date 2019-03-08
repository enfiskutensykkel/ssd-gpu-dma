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
 * Virtual address space mapping.
 */
struct va_map
{
    bool                    mapped;     // Is segment mapped into virtual address space?
    sci_map_t               md;         // SISCI mapping descriptor
};



/*
 * Local segment descriptor.
 * map.range.remote = false
 */
struct local_segment
{
    // XXX: ctrl reference can be replaced with a new sci_desc_t
    struct controller*      ctrl;       // Controller reference
    uint32_t                adapter;    // DIS adapter number
    sci_local_segment_t     segment;    // Local segment reference
    bool                    remove;     // Requires remove
    struct va_map           map;        // Mapping descriptor
    struct va_range         range;      // Memory range descriptor
};



/*
 * Remote segment descriptor.
 * map.range.remote = true
 */
struct remote_segment
{
    // XXX: ctrl reference is only necessary for device segments
    struct controller*      ctrl;       // Controller reference
    sci_remote_segment_t    segment;    // Remote segment reference
    bool                    disconnect; // Requires a disconnect
    struct va_map           map;        // Mapping descriptor
    struct va_range         range;      // Memory range descriptor
};


#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_MAP_H__ */
