#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "regs.h"

#ifndef NDEBUG
#include <stdio.h>
#endif

#ifdef _SISCI
#include "pcidev.h"
#include <sisci_types.h>
#include <sisci_api.h>
#include <sisci_error.h>
#endif /* #ifdef _SISCI */


/* Convenience defines */
#define encode_page_size(ps)        b2log((ps) >> 12)
#define encode_entry_size(es)       b2log(es)


void nvm_queue_clear(nvm_queue_t* queue, const nvm_ctrl_t* ctrl, int cq, uint16_t no, void* vaddr, uint64_t ioaddr)
{
    queue->no = no;
    queue->max_entries = 0;
    queue->entry_size = cq ? sizeof(nvm_cpl_t) : sizeof(nvm_cmd_t);
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 1;
    queue->vaddr = vaddr;
    queue->ioaddr = ioaddr;
    queue->db = cq ? CQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd) : SQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd);
    queue->max_entries = _MIN(ctrl->max_entries, ctrl->page_size / queue->entry_size);
}


/* Delay execution by 1 millisecond */
static inline uint64_t delay(uint64_t remaining)
{
    if (remaining == 0)
    {
        return 0;
    }

    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = _MIN(1000000UL, remaining);

    clock_nanosleep(CLOCK_REALTIME, 0, &ts, NULL);

    remaining -= _MIN(1000000UL, remaining);
    return remaining;
}


int nvm_ctrl_init_raw(nvm_ctrl_t* handle, volatile void* mm_ptr, size_t mm_size)
{
    nvm_ctrl_t ctrl;

    if (mm_size < NVM_CTRL_MEM_MINSIZE)
    {
        return EINVAL;
    }

    ctrl.dev_ref = NULL;
    ctrl.mm_size = mm_size;
    ctrl.mm_ptr = mm_ptr;

    // Retrieve host's page size
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to retrieve host page size: %s\n", strerror(errno));
#endif
        return errno;
    }

    uint8_t host_page_size = encode_page_size(page_size);
    uint8_t max_page_size = CAP$MPSMAX(ctrl.mm_ptr);
    uint8_t min_page_size = CAP$MPSMIN(ctrl.mm_ptr);

    if ( ! (min_page_size <= host_page_size && host_page_size <= max_page_size) )
    {
#ifndef NDEBUG
        fprintf(stderr, "Host is configured with an unsupported page size\n");
#endif
        return ERANGE;
    }

    // Set controller properties
    ctrl.page_size = page_size;
    ctrl.dstrd = CAP$DSTRD(ctrl.mm_ptr);
    ctrl.timeout = CAP$TO(ctrl.mm_ptr) * 500UL;
    ctrl.max_entries = CAP$MQES(ctrl.mm_ptr) + 1; // CAP.MQES is 0's based
    
    *handle = ctrl;
    return 0;
}


void nvm_ctrl_free(nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
#ifdef _SISCI
        if (ctrl->dev_ref != NULL)
        {
            pci_dev_ref_put(ctrl->dev_ref);
            free(ctrl->dev_ref);
            ctrl->dev_ref = NULL;
        }
#endif
    }
}


int nvm_ctrl_reset(const nvm_ctrl_t* ctrl, uint64_t acq_addr, uint64_t asq_addr)
{
    volatile uint32_t* cc = CC(ctrl->mm_ptr);

    // Set CC.EN to 0
    *cc = *cc & ~1;

    // Wait for CSTS.RDY to transition from 1 to 0
    uint64_t timeout = ctrl->timeout * 1000000UL;
    uint64_t remaining = delay(timeout);

    while (CSTS$RDY(ctrl->mm_ptr) != 0)
    {
        if (remaining == 0)
        {
#ifndef NDEBUG
            fprintf(stderr, "Timeout exceeded while waiting for reset\n");
#endif
            return ETIME;
        }

        remaining = delay(remaining);
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
    remaining = delay(timeout);

    while (CSTS$RDY(ctrl->mm_ptr) != 1)
    {
        if (remaining == 0)
        {
#ifndef NDEBUG
            fprintf(stderr, "Timeout exceeded while waiting for enable\n");
#endif
            return ETIME;
        }

        remaining = delay(remaining);
    }

    return 0;
}


#ifdef _SISCI

static int connect_device_bar(struct pci_dev* dev, uint64_t dev_id, uint32_t adapter, int bar, size_t size)
{
    int status;
    sci_error_t err;

    dev->device_id = dev_id;
    dev->adapter = adapter;
    dev->mm_size = size;

    SCIOpen(&dev->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create SISCI virtual device: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto quit;
    }

    SCIBorrowDevice(dev->sd, &dev->device, dev_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to increase device reference: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto close;
    }

    SCIConnectDeviceMemory(dev->sd, &dev->segment, dev->device, adapter, bar, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to connect to device memory: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto return_dev;
    }

    dev->mm_ptr = SCIMapRemoteSegment(dev->segment, &dev->map, 0, size, NULL, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to memory-map device memory: %s\n", SCIGetErrorString(err));
#endif
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
#ifndef NDEBUG
        fprintf(stderr, "Failed to disconnect from device memory: %s\n", SCIGetErrorString(err));
#endif
    }

return_dev:
    SCIReturnDevice(dev->device, 0, &err);

close:
    SCIClose(dev->sd, 0, &err);

quit:
    return status;
}


int pci_dev_ref_get(struct pci_dev* handle, uint64_t dev_id, uint32_t adapter, int bar, size_t size)
{
    int err;
    struct pci_dev dev;

    err = connect_device_bar(&dev, dev_id, adapter, bar, size);
    if (err != 0)
    {
        return err;
    }

    *handle = dev;
    return 0;
}


void pci_dev_ref_put(struct pci_dev* dev)
{
    sci_error_t err;

    dev->mm_ptr = NULL;
    dev->mm_size = 0;

    do
    {
        SCIUnmapSegment(dev->map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to unmap device memory: %s\n", SCIGetErrorString(err));
#endif
    }

    do
    {
        SCIDisconnectSegment(dev->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to disconnect from device memory: %s\n", SCIGetErrorString(err));
#endif
    }

    SCIReturnDevice(dev->device, 0, &err);
    SCIClose(dev->sd, 0, &err);
}


int nvm_ctrl_init(nvm_ctrl_t* handle, uint64_t device_id, uint32_t adapter)
{
    int err;
    nvm_ctrl_t ctrl;

    struct pci_dev* dev = (struct pci_dev*) malloc(sizeof(struct pci_dev));
    if (dev == NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to allocate device reference: %s\n", strerror(errno));
#endif
        return ENOMEM;
    }

    err = pci_dev_ref_get(dev, device_id, adapter, 0, NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        free(dev);
        return err;
    }

    err = nvm_ctrl_init_raw(&ctrl, dev->mm_ptr, dev->mm_size);
    if (err != 0)
    {
        free(dev);
        return err;
    }

    ctrl.dev_ref = dev;
    *handle = ctrl;
    return 0;
}


int nvm_dma_window_create(sci_local_segment_t segment, const nvm_ctrl_t* ctrl, uint64_t* ioaddr)
{
    sci_error_t err;
    sci_ioaddr_t addr;

    if (ctrl->dev_ref == NULL)
    {
        return EINVAL;
    }

    SCIMapSegmentForDevice(segment, ctrl->dev_ref->device, ctrl->dev_ref->adapter, &addr, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create DMA window for controller: %s\n", SCIGetErrorString(err));
#endif
        return EIO;
    }

    *ioaddr = (uint64_t) addr;
    return 0;
}


int nvm_dma_window_destroy(sci_local_segment_t segment, const nvm_ctrl_t* ctrl)
{
    sci_error_t err;

    if (ctrl->dev_ref == NULL)
    {
        return EINVAL;
    }

    SCIUnmapSegmentForDevice(segment, ctrl->dev_ref->device, ctrl->dev_ref->adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to destroy DMA window for controller: %s\n", SCIGetErrorString(err));
#endif
        return EIO;
    }

    return 0;
}

#endif /* #ifdef _SISCI */

