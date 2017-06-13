#include "memory.h"
#include "nvm/types.h"
#include "nvm/queue.h"
#include "nvm/command.h"
#include "nvm/util.h"
#include "nvm/ctrl.h"
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>


int workload(nvm_ctrl_t* ctrl, uint32_t ns, void* io_mem, size_t io_size)
{
    int err;
    buffer_t* sq_memory;
    buffer_t* cq_memory;
    nvm_queue_t q[2];

    sq_memory = get_buffer(-1, 20, ctrl->page_size, ctrl->page_size, ctrl->device_id);
    if (sq_memory == NULL)
    {
        return ENOMEM;
    }

    cq_memory = get_buffer(-1, 21, ctrl->page_size, ctrl->page_size, ctrl->device_id);
    if (cq_memory == NULL)
    {
        put_buffer(sq_memory);
        return ENOMEM;
    }

    err = nvm_create_cq(ctrl, &q[0], 1, cq_memory->virt_addr, cq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create completion queue: %s\n", strerror(err));
        put_buffer(sq_memory);
        put_buffer(cq_memory);
        return EIO;
    }

    err = nvm_create_sq(ctrl, &q[0], &q[1], 1, sq_memory->virt_addr, sq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create submission queue: %s\n", strerror(err));
        put_buffer(sq_memory);
        put_buffer(cq_memory);
        return EIO;
    }

    return 0;
}
