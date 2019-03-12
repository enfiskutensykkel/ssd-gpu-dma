#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include "mutex.h"
#include "ctrl.h"
#include "util.h"
#include "regs.h"
#include "dprintf.h"


/* Convenience defines */
#define encode_page_size(ps)    _nvm_b2log((ps) >> 12)
#define encode_entry_size(es)   _nvm_b2log(es)



/*
 * Helper function to allocate a handle container.
 */
static struct controller* create_handle(struct device* dev, const struct device_ops* ops, enum device_type type)
{
    int err;
    struct controller* handle;

    if (dev != NULL && (ops == NULL || ops->release_device == NULL))
    {
        dprintf("Inconsistent state, device operations is not set\n");
        return NULL;
    }

    handle = (struct controller*) malloc(sizeof(struct controller));
    if (handle == NULL)
    {
        dprintf("Failed to allocate controller handle: %s\n", strerror(errno));
        return NULL;
    }

    memset(&handle->handle, 0, sizeof(nvm_ctrl_t));
   
    err = _nvm_mutex_init(&handle->lock);
    if (err != 0)
    {
        free(handle);
        return NULL;
    }
    
    handle->count = 1;
    handle->device = dev;
    handle->type = type;
    if (ops != NULL)
    {
        handle->ops = *ops;
    }
    else
    {
        memset(&handle->ops, 0, sizeof(struct device_ops));
    }

    return handle;
}



static void remove_handle(struct controller* handle)
{
    int status;

    do
    {
        status = _nvm_mutex_free(&handle->lock);
    }
    while (status == EBUSY);

    free(handle);
}



/*
 * Take device reference.
 */
struct controller* _nvm_ctrl_get(const nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        // This is technically undefined behaviour (casting away const),
        // but we are not modifying the handle itself, only the container.
        struct controller* controller = _nvm_container_of(ctrl, struct controller, handle);

        int err = _nvm_mutex_lock(&controller->lock);
        if (err != 0)
        {
            dprintf("Failed to take device reference lock: %s\n", strerror(err));
            return NULL;
        }

        // Increase reference count
        ++controller->count;

        _nvm_mutex_unlock(&controller->lock);

        return controller;
    }

    return NULL;
}



/*
 * Release device reference.
 */
void _nvm_ctrl_put(struct controller* controller)
{
    if (controller != NULL)
    {
        uint32_t count = 0;

        _nvm_mutex_lock(&controller->lock);
        count = --controller->count;
        if (count == 0)
        {
            if (controller->device != NULL)
            {
                controller->ops.release_device(controller->device, controller->handle.mm_ptr, controller->handle.mm_size);
            }

            controller->device = NULL;
        }
        _nvm_mutex_unlock(&controller->lock);

        if (count == 0)
        {
            remove_handle(controller);
        }
    }
}



int _nvm_ctrl_init(nvm_ctrl_t** handle, struct device* dev, const struct device_ops* ops, enum device_type type,
        volatile void* mm_ptr, size_t mm_size)
{
    struct controller* container;
    nvm_ctrl_t* ctrl;

    *handle = NULL;

    container = create_handle(dev, ops, type);
    if (container == NULL)
    {
        return ENOMEM;
    }

    ctrl = &container->handle;
    ctrl->mm_ptr = mm_ptr;
    ctrl->mm_size = mm_size;

    if (ctrl->mm_size < NVM_CTRL_MEM_MINSIZE)
    {
        remove_handle(container);
        return EINVAL;
    }

    // Get the system page size
    size_t page_size = _nvm_host_page_size();
    if (page_size == 0)
    {
        remove_handle(container);
        return ENOMEM;
    }

    // Get the controller page size
    uint8_t host_page_size = encode_page_size(page_size);
    uint8_t max_page_size = CAP$MPSMAX(ctrl->mm_ptr);
    uint8_t min_page_size = CAP$MPSMIN(ctrl->mm_ptr);

    if ( ! (min_page_size <= host_page_size && host_page_size <= max_page_size) )
    {
        remove_handle(container);
        dprintf("System page size is incompatible with controller page size\n");
        return ERANGE;
    }

    // Set controller properties
    ctrl->page_size = page_size;
    ctrl->dstrd = CAP$DSTRD(ctrl->mm_ptr);
    ctrl->timeout = CAP$TO(ctrl->mm_ptr) * 500UL;
    ctrl->max_qs = CAP$MQES(ctrl->mm_ptr) + 1; // CAP.MQES is 0's based

    *handle = ctrl;

    return 0;
}



/*
 * Reset the controller manually.
 * We deliberately use the pointers from the user-handle, in case the user
 * overrides them somehow.
 */
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
    *aqa = AQA$ACQS(cq_max_entries) | AQA$ASQS(sq_max_entries);
    
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



void nvm_ctrl_free(nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        struct controller* container = _nvm_container_of(ctrl, struct controller, handle);
        _nvm_ctrl_put(container);
    }
}



int nvm_raw_ctrl_init(nvm_ctrl_t** ctrl, volatile void* mm_ptr, size_t mm_size)
{
    return _nvm_ctrl_init(ctrl, NULL, NULL, DEVICE_TYPE_UNKNOWN, mm_ptr, mm_size);
}

