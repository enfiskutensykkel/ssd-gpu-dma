#ifdef _SISCI
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include "dis/device.h"
#include "dis/map.h"
#include "ctrl.h"
#include "util.h"
#include "regs.h"
#include "dprintf.h"



/* Forward declaration */
struct memory_reference;


/*
 * Controller device type.
 * Indicates how we access the controller.
 */
enum device_type
{
    _DEVICE_TYPE_UNKNOWN        = 0x00, // Device is mapped manually by the user
    _DEVICE_TYPE_SYSFS          = 0x01, // Device is mapped through file descriptor
    _DEVICE_TYPE_SMARTIO        = 0x04  // Device is mapped by SISCI SmartIO API
};



/*
 * Internal handle container.
 */
struct controller
{
    enum device_type            type;   // Device type
    struct memory_reference*    ref;    // Reference to mapped BAR0
    int                         fd;     // File descriptor to memory mapping
    nvm_ctrl_t                  handle; // User handle
};


#ifdef _SISCI
struct memory_reference
{
    struct device_memory        bar;    // Reference to BAR0
    struct va_map               map;    // Mapping descriptor
};
#endif




/* Convenience defines */
#define encode_page_size(ps)    _nvm_b2log((ps) >> 12)
#define encode_entry_size(es)   _nvm_b2log(es)

#define container(ctrl)         \
    ((struct controller*) (((unsigned char*) (ctrl)) - offsetof(struct controller, handle)))

#define const_container(ctrl)   \
    ((const struct controller*) (((const unsigned char*) (ctrl)) - offsetof(struct controller, handle)))





/*
 * Look up file descriptor from controller handle.
 */
int _nvm_fd_from_ctrl(const nvm_ctrl_t* ctrl)
{
    const struct controller* container = const_container(ctrl);

    switch (container->type)
    {
        case _DEVICE_TYPE_SYSFS:
            return container->fd;

        default:
            return -EBADF;
    }
}



#ifdef _SISCI
/*
 * Look up device from controller handle.
 */
const struct device* _nvm_device_from_ctrl(const nvm_ctrl_t* ctrl)
{
    const struct controller* container = const_container(ctrl);

    if (container->type == _DEVICE_TYPE_SMARTIO && container->ref != NULL)
    {
        return &container->ref->bar.device;
    }

    return NULL;
}
#endif



/*
 * Helper function to allocate a handle container.
 */
static struct controller* create_container()
{
    struct controller* container = (struct controller*) malloc(sizeof(struct controller));
    
    if (container == NULL)
    {
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        return NULL;
    }

    container->type = _DEVICE_TYPE_UNKNOWN;
    container->fd = -1;
    container->ref = NULL;

    return container;
}



#ifdef _SISCI
/*
 * Helper function to increase a device reference and connect 
 * to a PCI BAR0 on the controller's device.
 */
static int connect_register_memory(struct memory_reference** ref, const struct device* dev, uint32_t adapter)
{
    *ref = NULL;

    struct memory_reference* mem = (struct memory_reference*) malloc(sizeof(struct memory_reference));
    if (mem == NULL)
    {
        dprintf("Failed to allocate controller memory reference: %s\n", strerror(errno));
        return ENOMEM;
    }

    int err = _nvm_device_memory_get(&mem->bar, dev, adapter, 0, SCI_FLAG_BAR);
    if (err != 0)
    {
        free(mem);
        dprintf("Failed to get controller memory reference: %s\n", strerror(err));
        return err;
    }

    err = _nvm_va_map_remote(&mem->map, NVM_CTRL_MEM_MINSIZE, mem->bar.segment, true, false);
    if (err != 0)
    {
        _nvm_device_memory_put(&mem->bar);
        free(mem);
        dprintf("Failed to map controller memory: %s\n", strerror(err));
        return err;
    }

    *ref = mem;
    return 0;
}
#endif



#ifdef _SISCI
static void disconnect_register_memory(struct memory_reference* ref)
{
    _nvm_va_unmap(&ref->map);
    _nvm_device_memory_put(&ref->bar);
    free(ref);
}
#endif



/* 
 * Helper function to initialize the controller handle by reading
 * the appropriate registers from the controller BAR.
 */
static int initialize_handle(nvm_ctrl_t* ctrl, volatile void* mm_ptr, size_t mm_size)
{
    if (mm_size < NVM_CTRL_MEM_MINSIZE)
    {
        return EINVAL;
    }

    ctrl->mm_size = mm_size;
    ctrl->mm_ptr = mm_ptr;

    // Get the system page size
    size_t page_size = _nvm_host_page_size();
    if (page_size == 0)
    {
        return ENOMEM;
    }

    // Get the controller page size
    uint8_t host_page_size = encode_page_size(page_size);
    uint8_t max_page_size = CAP$MPSMAX(mm_ptr);
    uint8_t min_page_size = CAP$MPSMIN(mm_ptr);

    if ( ! (min_page_size <= host_page_size && host_page_size <= max_page_size) )
    {
        dprintf("System page size is incompatible with controller page size\n");
        return ERANGE;
    }

    // Set controller properties
    ctrl->page_size = page_size;
    ctrl->dstrd = CAP$DSTRD(mm_ptr);
    ctrl->timeout = CAP$TO(mm_ptr) * 500UL;
    ctrl->max_entries = CAP$MQES(mm_ptr) + 1; // CAP.MQES is 0's based

    return 0;
}



int nvm_raw_ctrl_reset(const nvm_ctrl_t* ctrl, uint64_t acq_addr, uint64_t asq_addr)
{
    volatile uint32_t* cc = CC(ctrl->mm_ptr);

    // Set CC.EN to 0
    *cc = *cc & ~1;

    // Wait for CSTS.RDY to transition from 1 to 0
    uint64_t timeout = ctrl->timeout * 1000000UL;
    uint64_t remaining = _nvm_delay_remain(timeout);

    while (CSTS$RDY(ctrl->mm_ptr) != 0)
    {
        if (remaining == 0)
        {
            dprintf("Timeout exceeded while waiting for controller reset\n");
            return ETIME;
        }

        remaining = _nvm_delay_remain(remaining);
    }

    // Set admin queue attributes
    volatile uint32_t* aqa = AQA(ctrl->mm_ptr);

    uint32_t cq_max_entries = ctrl->page_size / sizeof(nvm_cpl_t) - 1;
    uint32_t sq_max_entries = ctrl->page_size / sizeof(nvm_cmd_t) - 1;
    *aqa = AQA$AQS(sq_max_entries) | AQA$AQC(cq_max_entries);
    
    // Set admin completion queue
    volatile uint64_t* acq = ACQ(ctrl->mm_ptr);
    *acq = acq_addr;

    // Set admin submission queue
    volatile uint64_t* asq = ASQ(ctrl->mm_ptr);
    *asq = asq_addr;

    // Set CC.MPS to pagesize and CC.EN to 1
    uint32_t cqes = encode_entry_size(sizeof(nvm_cpl_t)); 
    uint32_t sqes = encode_entry_size(sizeof(nvm_cmd_t)); 
    *cc = CC$IOCQES(cqes) | CC$IOSQES(sqes) | CC$MPS(encode_page_size(ctrl->page_size)) | CC$CSS(0) | CC$EN(1);

    // Wait for CSTS.RDY to transition from 0 to 1
    remaining = _nvm_delay_remain(timeout);

    while (CSTS$RDY(ctrl->mm_ptr) != 1)
    {
        if (remaining == 0)
        {
            dprintf("Timeout exceeded while waiting for controller enable\n");
            return ETIME;
        }

        remaining = _nvm_delay_remain(remaining);
    }

    return 0;
}



int nvm_raw_ctrl_init(nvm_ctrl_t** ctrl, volatile void* mm_ptr, size_t mm_size)
{
    int err;
    *ctrl = NULL;

    struct controller* container = create_container();
    if (container == NULL)
    {
        return ENOMEM;
    }

    container->type = _DEVICE_TYPE_UNKNOWN;

    err = initialize_handle(&container->handle, mm_ptr, mm_size);
    if (err != 0)
    {
        free(container);
        return err;
    }

    *ctrl = &container->handle;
    return 0;
}



#ifdef _SISCI
int nvm_dis_ctrl_init(nvm_ctrl_t** ctrl, uint64_t dev_id, uint32_t adapter)
{
    int err;
    *ctrl = NULL;

    struct controller* container = create_container();
    if (container == NULL)
    {
        return ENOMEM;
    }

    container->type = _DEVICE_TYPE_SMARTIO;

    struct device dev;
    err = _nvm_device_get(&dev, dev_id);
    if (err != 0)
    {
        free(container);
        return err;
    }

    err = connect_register_memory(&container->ref, &dev, adapter);
    if (err != 0)
    {
        _nvm_device_put(&dev);
        free(container);
        return err;
    }

    _nvm_device_put(&dev);

    size_t size = SCIGetRemoteSegmentSize(container->ref->bar.segment);

    err = initialize_handle(&container->handle, container->ref->map.vaddr, size);
    if (err != 0)
    {
        disconnect_register_memory(container->ref);
        free(container);
        return err;
    }

    *ctrl = &container->handle;
    return 0;
}
#endif



#if defined ( __unix__ )
int nvm_ctrl_init(nvm_ctrl_t** ctrl, int filedes)
{
    int err;
    
    int fd = dup(filedes);
    if (fd < 0)
    {
        dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
        return errno;
    }

    err = fcntl(fd, F_SETFD, O_RDWR|O_NONBLOCK);
    if (err == -1)
    {
        close(fd);
        dprintf("Failed to set file descriptor flags: %s\n", strerror(errno));
        return errno;
    }

    volatile void* ptr = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE, fd, 0);
    if (ptr == NULL)
    {
        close(fd);
        dprintf("Failed to map device memory: %s\n", strerror(errno));
        return errno;
    }

    struct controller* container = create_container();
    if (container == NULL)
    {
        munmap((void*) ptr, NVM_CTRL_MEM_MINSIZE);
        close(fd);
        return ENOMEM;
    }

    container->type = _DEVICE_TYPE_SYSFS;

    err = initialize_handle(&container->handle, ptr, NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        munmap((void*) ptr, NVM_CTRL_MEM_MINSIZE);
        free(container);
        close(fd);
        return err;
    }

    container->fd = fd;

    *ctrl = &container->handle;
    return 0;
}
#endif



void nvm_ctrl_free(nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        struct controller* container = container(ctrl);

        switch (container->type)
        {
            case _DEVICE_TYPE_UNKNOWN:
                // Do nothing
                break;

#if defined ( __unix__ )
            case _DEVICE_TYPE_SYSFS:
                munmap((void*) ctrl->mm_ptr, ctrl->mm_size);
                close(container->fd);
                break;
#endif

#if _SISCI
            case _DEVICE_TYPE_SMARTIO:
                disconnect_register_memory(container->ref);
                break;
#endif

            default:
                dprintf("Unknown controller type\n");
                break;
        }

        free(container);
    }
}



#ifdef _SISCI
void nvm_dis_ctrl_unmap_p2p_device(const nvm_ctrl_t* ctrl, sci_device_t dev)
{
    if (ctrl != NULL)
    {
        const struct controller* container = const_container(ctrl);

        if (container->type == _DEVICE_TYPE_SMARTIO && container->ref != NULL)
        {
            sci_error_t err;
            do
            {
                SCIUnmapRemoteSegmentForDevice(container->ref->bar.segment, dev, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
        }
    }
}
#endif



#ifdef _SISCI
int nvm_dis_ctrl_map_p2p_device(const nvm_ctrl_t* ctrl, sci_device_t dev, uint64_t* ioaddr)
{
    if (ctrl == NULL)
    {
        return EINVAL;
    }

    const struct controller* container = const_container(ctrl);

    if (container->type != _DEVICE_TYPE_SMARTIO || container->ref == NULL)
    {
        return EINVAL;
    }

    sci_error_t err;
    sci_ioaddr_t addr;
    SCIMapRemoteSegmentForDevice(container->ref->bar.segment, dev, &addr, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map controller BAR for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    if (ioaddr != NULL)
    {
        *ioaddr = addr;
    }
    return 0;
}
#endif

