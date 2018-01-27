#ifndef __NVM_INTERNAL_DIS_MAPPING_H__
#define __NVM_INTERNAL_DIS_MAPPING_H__

/* Forward declarations */
struct va_map;
struct io_map;
struct device;
struct device_memory;
struct local_memory;


#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <sisci_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>



/*
 * VA mapping kind.
 * Indicates whether or not a mapping is in use.
 */
enum va_map_type
{
    _VA_MAP_NOT_MAPPED      = 0x00,
    _VA_MAP_MAPPED_LOCAL    = 0x01,
    _VA_MAP_MAPPED_REMOTE   = 0x02
};

/*
 * Virtual address mapping descriptor.
 *
 * Describes a mapping of a segment in local virtual address space.
 */
struct va_map
{
    enum va_map_type        type;           // Type of mapping
    sci_map_t               md;             // SISCI mapping descriptor
    volatile void*          vaddr;          // Pointer to mapped memory
};

#define VA_MAP_CLEAR(p) \
    do { (p)->type = _VA_MAP_NOT_MAPPED; (p)->vaddr = NULL; } while (0)

#define VA_MAP_INIT     { .type = _VA_MAP_NOT_MAPPED, .md = NULL, .vaddr = NULL }



/*
 * IO mapping kind.
 * Indicates whether or not a mapping is in use.
 */
enum io_map_type
{
    _IO_MAP_NOT_MAPPED      = 0x00,
    _IO_MAP_MAPPED_REMOTE   = 0x01,
    _IO_MAP_MAPPED_LOCAL    = 0x02,
};

/*
 * IO address space mapping descriptor.
 *
 * Describes a mapping of a segment as seen from the device.
 */
struct io_map
{
    enum io_map_type        type;           // Type of mapping
    sci_local_segment_t     lsegment;       // Local segment descriptor
    sci_remote_segment_t    rsegment;       // Remote segment descriptor
    sci_device_t            device;         // SmartIO device the segment is mapped for
    uint32_t                adapter;        // Local adapter
    sci_ioaddr_t            ioaddr;         // Device-local address to segment
};

#define IO_MAP_CLEAR(p) \
    do { (p)->type = _IO_MAP_NOT_MAPPED; (p)->ioaddr = 0; } while (0)

#define IO_MAP_INIT     \
    { .type = _IO_MAP_NOT_MAPPED, .lsegment = NULL, .rsegment = NULL, .device = NULL, .adapter = 0, .ioaddr = 0 }





/*
 * Map local segment into virtual address space.
 */
int _nvm_va_map_local(struct va_map* map, size_t size, sci_local_segment_t segment);


/*
 * Map remote segment into virtual address space.
 */
int _nvm_va_map_remote(struct va_map* map, 
                       size_t size, 
                       sci_remote_segment_t segment, 
                       bool write, 
                       bool wc);


/*
 * Unmap segment from virtual address space.
 */
void _nvm_va_unmap(struct va_map* map);



/*
 * Map local segment for device.
 */
int _nvm_io_map_local(struct io_map* map, 
                      sci_device_t device, 
                      sci_local_segment_t segment,
                      uint32_t adapter);


/*
 * Map remote segment for device.
 */
int _nvm_io_map_remote(struct io_map* map,
                       sci_device_t device,
                       sci_remote_segment_t segment);


/*
 * Unmap segment for device.
 */
void _nvm_io_unmap(struct io_map* map);



/*
 * Convenience function to map device memory for device.
 */
int _nvm_io_map_device_memory(struct io_map* map, const struct device_memory* mem);



/*
 * Convenience function to map local segment for device.
 * This will implicitly prepare segment on the specified adapter.
 */
int _nvm_io_map_local_memory(struct io_map* map, 
                             const struct device* dev, 
                             const struct local_memory* mem,
                             uint32_t adapter);



#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_MAPPING_H__ */
