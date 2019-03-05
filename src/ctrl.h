#ifndef __NVM_INTERNAL_CTRL_H__
#define __NVM_INTERNAL_CTRL_H__

#include <nvm_types.h>
#include "mutex.h"
#include "util.h"


/* 
 * Device handle.
 */
struct device;



/*
 * Forward declaration of a virtual memory address range.
 */
struct va_range;



/*
 * Device reference operations.
 */
struct device_ops
{
    /*
     * Release device reference (called when refcount is 0)
     * This should also unmap MLBAR/BAR0 of the device.
     */
    void (*release_device)(struct device* dev, volatile void* mm_ptr, size_t mm_size);


    /*
     * Map an address range for the device.
     */
    int (*map_range)(const struct device* dev, const struct va_range* va, uint64_t* ioaddrs);

    
    /*
     * Unmap an address range for the device.
     */
    void (*unmap_range)(const struct device* dev, const struct va_range* va);
};



/*
 * Controller device type.
 * Indicates how the controller handle was initialized.
 */
enum device_type
{
    DEVICE_TYPE_UNKNOWN =   0x00,       /* Device is mapped manually by the user */
    DEVICE_TYPE_IOCTL   =   0x01,       /* Device is mapped through UNIX file descriptor */
    DEVICE_TYPE_SMARTIO =   0x02,       /* Device is mapped by SISCI SmartIO API */
};



/*
 * Internal controller handle.
 * Used to reference count the device handle.
 * Reference counting is handled by the get and put functions.
 */
struct controller
{
    struct mutex                lock;           /* Ensure exclusive access */
    uint32_t                    count;          /* Reference count */
    enum device_type            type;           /* Controller device type */
    struct device*              device;         /* Device handle */
    struct device_ops           ops;            /* Device operations */
    nvm_ctrl_t                  handle;         /* User's handle */
};


/* 
 * Helper function to initialize the controller handle by reading
 * the appropriate registers from the controller BAR.
 */
int _nvm_ctrl_init(nvm_ctrl_t** handle,             /* User's handle */
                   struct device* dev,              /* Device handle */
                   const struct device_ops* ops,    /* Device handle operations */
                   enum device_type type,           /* Device type */
                   volatile void* mm_ptr,           /* Memory-mapped pointer */
                   size_t mm_size);                 /* Size of memory-map */



/*
 * Increase controller reference count.
 */
struct controller* _nvm_ctrl_get(const nvm_ctrl_t* handle);



/*
 * Decrease controller reference count.
 */
void _nvm_ctrl_put(struct controller* ctrl);



/*
 * Convenience macro to get the controller type.
 */
#define _nvm_ctrl_type(ctrl) _nvm_container_of(ctrl, struct controller, handle)->type

#endif /* __NVM_INTERNAL_CTRL_H__ */
