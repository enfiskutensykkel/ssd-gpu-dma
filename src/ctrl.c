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
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "device.h"
#include "util.h"
#include "regs.h"
#include "dprintf.h"


/* Forward declaration */
struct bar_reference;


#ifdef _SISCI
/*
 * Reference to a PCI device in the cluster.
 *
 * This structure is used to hold a SmartIO reference to the physical 
 * controller.
 */
struct bar_reference
{
    struct nvm_device       device;         // Device reference
    sci_remote_segment_t    segment;        // SISCI remote segment to device BAR
    sci_map_t               map;            // SISCI memory map handle
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to device BAR
};
#endif


/*
 * NVM controller handle container.
 */
struct handle_container
{
    int                         ioctl_fd;   // File descriptor
    struct bar_reference*       bar_ref;    // Device BAR reference handle
    struct nvm_controller       ctrl;       // Actual controlle handle
};



/* Convenience defines */
#define encode_page_size(ps)        _nvm_b2log((ps) >> 12)
#define encode_entry_size(es)       _nvm_b2log(es)



/*
 * Helper function to retrieve a controller handle's surrounding container.
 */
static inline struct handle_container* get_container(nvm_ctrl_t ctrl)
{
    return (struct handle_container*) (((unsigned char*) ctrl) - offsetof(struct handle_container, ctrl));
}



/*
 * Helper function to retrieve a controller handle's surrounding container.
 */
static inline const struct handle_container* get_container_const(const struct nvm_controller* ctrl)
{
    return (const struct handle_container*) (((const unsigned char*) ctrl) - offsetof(struct handle_container, ctrl));
}



int nvm_ctrl_reset(nvm_ctrl_t ctrl, uint64_t acq_addr, uint64_t asq_addr)
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
    uint32_t cq_max_entries = (ctrl->page_size / sizeof(nvm_cpl_t)) - 1;
    uint32_t sq_max_entries = (ctrl->page_size / sizeof(nvm_cmd_t)) - 1;
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



/* 
 * Helper function to initialize the controller handle by reading
 * the appropriate registers from the controller BAR.
 */
static int initialize_handle(nvm_ctrl_t ctrl, volatile void* mm_ptr, size_t mm_size)
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



int nvm_ctrl_init_userspace(nvm_ctrl_t* ctrl_handle, volatile void* mm_ptr, size_t mm_size)
{
    int err;

    *ctrl_handle = NULL;

    struct handle_container* container = (struct handle_container*) malloc(sizeof(struct handle_container));
    if (container == NULL)
    {
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    container->ioctl_fd = -1;
    container->bar_ref = NULL;

    err = initialize_handle(&container->ctrl, mm_ptr, mm_size);
    if (err != 0)
    {
        free(container);
        return err;
    }

    *ctrl_handle = &container->ctrl;
    return 0;
}



int nvm_ctrl_init(nvm_ctrl_t* ctrl_handle, uint64_t device_id)
{
    int err;
    char path[128];

    *ctrl_handle = NULL;

    snprintf(path, sizeof(path), "/dev/disnvme%lu", device_id);
    int fd = open(path, O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        dprintf("Could not find device %lu: %s\n", device_id, strerror(errno));
        return ENODEV;
    }

    volatile void* ptr = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE, fd, 0);
    if (ptr == NULL)
    {
        close(fd);
        dprintf("Failed to map BAR resource: %s\n", strerror(errno));
        return EIO;
    }

    struct handle_container* container = (struct handle_container*) malloc(sizeof(struct handle_container));
    if (container == NULL)
    {
        munmap((void*) ptr, NVM_CTRL_MEM_MINSIZE);
        close(fd);
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    container->ioctl_fd = fd;
    container->bar_ref = NULL;

    err = initialize_handle(&container->ctrl, ptr, NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        munmap((void*) ptr, NVM_CTRL_MEM_MINSIZE);
        close(fd);
        free(container);
        return err;
    }

    *ctrl_handle = &container->ctrl;
    return 0;
}



/*
 * Get the IOCTL file descriptor from the controller reference.
 */
int _nvm_ioctl_fd_from_ctrl(const struct nvm_controller* ctrl)
{
    const struct handle_container* container = get_container_const(ctrl);
    
    return container->ioctl_fd;
}



#ifdef _SISCI

/*
 * Look up device reference from a controller handle.
 */
const struct nvm_device* _nvm_dev_from_ctrl(const struct nvm_controller* ctrl)
{
    const struct handle_container* container = get_container_const(ctrl);

    if (container->bar_ref != NULL)
    {
        return &container->bar_ref->device;
    }

    return NULL;
}



/* 
 * Acquire a device reference.
 */
int _nvm_dev_get(struct nvm_device* dev, uint64_t dev_id, uint32_t adapter)
{
    sci_error_t err;

    dev->device_id = dev_id;
    dev->adapter = adapter;

    SCIOpen(&dev->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create virtual device: %s\n", SCIGetErrorString(err));
        return EIO;
    }

    SCIBorrowDevice(dev->sd, &dev->device, dev_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to increase device reference: %s\n", SCIGetErrorString(err));
        SCIClose(dev->sd, 0, &err);
        return ENODEV;
    }
    
    return 0;
}



/*
 * Release device reference.
 */
void _nvm_dev_put(struct nvm_device* dev)
{
    sci_error_t err;
    SCIReturnDevice(dev->device, 0, &err);
    SCIClose(dev->sd, 0, &err);
}


/* 
 * Connect and memory-map to a BAR region on a PCI device in the cluster 
 */
static int connect_device_bar(struct bar_reference* dev, int bar, size_t size)
{
    int status;
    sci_error_t err;

    dev->mm_size = size;

    SCIConnectDeviceMemory(dev->device.sd, &dev->segment, dev->device.device, dev->device.adapter, bar, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to connect to device memory: %s\n", SCIGetErrorString(err));
        status = ENODEV;
        goto quit;
    }

    dev->mm_ptr = SCIMapRemoteSegment(dev->segment, &dev->map, 0, size, NULL, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to memory-map device memory: %s\n", SCIGetErrorString(err));
        status = EIO;
        goto disconnect;
    }

    return 0;

disconnect:
    do
    {
        SCIDisconnectSegment(dev->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
    
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to disconnect from device memory: %s\n", SCIGetErrorString(err));
    }

quit:
    return status;
}



/* 
 * Acquire a reference to a device BAR region.
 */
static int get_bar_reference(struct bar_reference* ref, uint64_t dev_id, uint32_t adapter, int bar, size_t size)
{
    int err;

    err = _nvm_dev_get(&ref->device, dev_id, adapter);
    if (err != 0)
    {
        return err;
    }

    err = connect_device_bar(ref, bar, size);
    if (err != 0)
    {
        _nvm_dev_put(&ref->device);
        return err;
    }

    return 0;
}



/*
 * Release device BAR reference.
 */
static void put_bar_reference(struct bar_reference* bar)
{
    sci_error_t err;

    bar->mm_ptr = NULL;
    bar->mm_size = 0;

    do
    {
        SCIUnmapSegment(bar->map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to unmap device memory: %s\n", SCIGetErrorString(err));
    }

    do
    {
        SCIDisconnectSegment(bar->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to disconnect from device memory: %s\n", SCIGetErrorString(err));
    }

    _nvm_dev_put(&bar->device);
}



int nvm_dis_ctrl_init(nvm_ctrl_t* ctrl_handle, uint64_t device_id, uint32_t adapter)
{
    int err;

    *ctrl_handle = NULL;

    struct bar_reference* bar_ref = (struct bar_reference*) malloc(sizeof(struct bar_reference));
    if (bar_ref == NULL)
    {
        dprintf("Failed to allocate device reference handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    err = get_bar_reference(bar_ref, device_id, adapter, 0, NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        free(bar_ref);
        return err;
    }

    struct handle_container* container = (struct handle_container*) malloc(sizeof(struct handle_container));
    if (container == NULL)
    {
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        free(bar_ref);
        return ENOMEM;
    }

    container->ioctl_fd = -1;
    container->bar_ref = bar_ref;

    err = initialize_handle(&container->ctrl, bar_ref->mm_ptr, bar_ref->mm_size);
    if (err != 0)
    {
        free(bar_ref);
        free(container);
        return err;
    }

    *ctrl_handle = &container->ctrl;
    return 0;
}

#endif /* _SISCI */



void nvm_ctrl_free(nvm_ctrl_t ctrl)
{
    if (ctrl != NULL)
    {
        struct handle_container* container = get_container(ctrl);

        if (container->ioctl_fd >= 0)
        {

            munmap((void*) ctrl->mm_ptr, ctrl->mm_size);
            close(container->ioctl_fd);
        }

#if _SISCI
        if (container->bar_ref != NULL)
        {
            put_bar_reference(container->bar_ref);
            free(container->bar_ref);
        }
#endif

        free(container);
    }
}

