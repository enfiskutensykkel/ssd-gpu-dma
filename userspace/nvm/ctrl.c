#include "types.h"
#include "util.h"
#include "queue.h"
#include "command.h"
#include "ctrl.h"
#include "memory.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>


/* Controller registers */
#define CAP(p)          _REG(p, 0x0000, 64)     // Controller Capabilities
#define CC(p)           _REG(p, 0x0014, 32)     // Controller Configuration
#define CSTS(p)         _REG(p, 0x001c, 32)     // Controller Status
#define AQA(p)          _REG(p, 0x0024, 32)     // Admin Queue Attributes
#define ASQ(p)          _REG(p, 0x0028, 64)     // Admin Submission Queue Base Address
#define ACQ(p)          _REG(p, 0x0030, 64)     // Admin Completion Queue Base Address


/* Read bit fields */
#define CAP$MPSMAX(p)   _RB(*CAP(p), 55, 52)    // Memory Page Size Maximum
#define CAP$MPSMIN(p)   _RB(*CAP(p), 51, 48)    // Memory Page Size Minimum
#define CAP$DSTRD(p)    _RB(*CAP(p), 35, 32)    // Doorbell Stride
#define CAP$TO(p)       _RB(*CAP(p), 31, 24)    // Timeout
#define CAP$CQR(p)      _RB(*CAP(p), 16, 16)    // Contiguous Queues Required
#define CAP$MQES(p)     _RB(*CAP(p), 15,  0)    // Maximum Queue Entries Supported

#define CSTS$RDY(p)     _RB(*CSTS(p), 0,  0)    // Ready indicator


/* Write bit fields */
#define CC$IOCQES(v)    _WB(v, 23, 20)          // IO Completion Queue Entry Size
#define CC$IOSQES(v)    _WB(v, 19, 16)          // IO Submission Queue Entry Size
#define CC$MPS(v)       _WB(v, 10,  7)          // Memory Page Size
#define CC$CSS(v)       _WB(0,  3,  1)          // IO Command Set Selected (0=NVM Command Set)
#define CC$EN(v)        _WB(v,  0,  0)          // Enable

#define AQA$AQS(v)      _WB(v, 27, 16)          // Admin Completion Queue Size
#define AQA$AQC(v)      _WB(v, 11,  0)          // Admin Submission Queue Size


/* List of admin command opcodes */
enum admin_command_set
{
    ADMIN_DELETE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x00,
    ADMIN_CREATE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x01,
    ADMIN_DELETE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x00,
    ADMIN_CREATE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x01,
    ADMIN_IDENTIFY_CONTROLLER       = (0x00 << 7) | (0x01 << 2) | 0x02,
    ADMIN_ABORT                     = (0x00 << 7) | (0x02 << 2) | 0x00,
    ADMIN_SET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x01,
    ADMIN_GET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x02
};


#define encode_page_size(page_size) b2log((page_size) >> 12)

#define encode_entry_size(entry_size) b2log(entry_size)


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


static int reset_controller(const nvm_ctrl_t* ctrl)
{
    volatile uint32_t* cc = CC(ctrl->reg_ptr);

    // Set CC.EN to 0
    *cc = *cc & ~1;

    // Wait for CSTS.RDY to transition from 1 to 0
    uint64_t timeout = ctrl->timeout * 1000000UL;
    uint64_t remaining = delay(timeout);

    while (CSTS$RDY(ctrl->reg_ptr) != 0)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 1 -> 0\n");
            remaining = timeout;
            return ETIME;
        }

        remaining = delay(remaining);
    }

    return 0;
}


static int enable_controller(volatile void* register_ptr, uint8_t encoded_page_size, uint64_t timeout)
{
    volatile uint32_t* cc = CC(register_ptr);

    // Set CC.MPS = <pagesize> and CC.EN to 1 
    *cc = CC$IOCQES(0) | CC$IOSQES(0) | CC$MPS(encoded_page_size) | CC$CSS(0) | CC$EN(1);

    // Wait for CSTS.RDY to transition from 0 to 1
    timeout = timeout * 1000000UL;
    uint64_t remaining = delay(timeout);

    while (CSTS$RDY(register_ptr) != 1)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 0 -> 1\n");
            remaining = timeout;
            return ETIME;
        }

        remaining = delay(remaining);
    }

    return 0;
}


static void configure_entry_sizes(const nvm_ctrl_t* controller)
{
    volatile uint32_t* cc = CC(controller->reg_ptr);

    uint32_t cqes = encode_entry_size(controller->cq_entry_size);
    uint32_t sqes = encode_entry_size(controller->sq_entry_size);

    *cc |= CC$IOCQES(cqes) | CC$IOSQES(sqes);
}


/* Set admin queue registers */
static void configure_admin_queues(nvm_ctrl_t* controller)
{
    nvm_queue_t* sq = &controller->admin_sq;
    nvm_queue_t* cq = &controller->admin_cq;

    sq->virt_addr = controller->admin_sq_page.virt_addr;
    sq->bus_addr = controller->admin_sq_page.bus_addr;
    sq->max_entries = controller->admin_sq_page.page_size / controller->sq_entry_size;
    memset(controller->admin_sq_page.virt_addr, 0, controller->admin_sq_page.page_size);

    cq->virt_addr = controller->admin_cq_page.virt_addr;
    cq->bus_addr = controller->admin_cq_page.bus_addr;
    cq->max_entries = controller->admin_cq_page.page_size / controller->cq_entry_size;
    memset(controller->admin_cq_page.virt_addr, 0, controller->admin_cq_page.page_size);

    volatile uint32_t* aqa = AQA(controller->reg_ptr);
    *aqa = AQA$AQS((uint32_t) sq->max_entries - 1) | AQA$AQC((uint32_t) cq->max_entries - 1);

    volatile uint64_t* asq = ASQ(controller->reg_ptr);
    *asq = sq->bus_addr;

    volatile uint64_t* acq = ACQ(controller->reg_ptr);
    *acq = cq->bus_addr;

    fprintf(stderr, "asq:%lx acq:%lx\n", sq->bus_addr, cq->bus_addr);
}


static int identify_controller(nvm_ctrl_t* controller)
{
    struct command* identify = sq_enqueue(&controller->admin_sq);
    if (identify  == NULL)
    {
        return ENOSPC;
    }

    cmd_header(identify, ADMIN_IDENTIFY_CONTROLLER, 0);
    cmd_data_ptr(identify, controller->identify.bus_addr, 0);
    identify->dword[10] = (0 << 16) | 0x01;

    struct command* get_features = sq_enqueue(&controller->admin_sq);
    if (get_features == NULL)
    {
        return ENOSPC;
    }

    uint16_t expected_id = *CMD_CID(get_features);
    cmd_header(get_features, ADMIN_GET_FEATURES, 0);

    get_features->dword[10] = (0x03 << 8) | 0x07;
    get_features->dword[11] = 0;

    // Wait for completions
    sq_submit(&controller->admin_sq);
    for (size_t i = 0; i < 2; ++i)
    {
        struct completion* cpl = cq_dequeue_block(&controller->admin_cq, controller->timeout);
        if (cpl == NULL)
        {
            fprintf(stderr, "Waiting for completion timed out\n");
            return ETIME;
        }

        sq_update(&controller->admin_sq, cpl);

        if (SCT(cpl) != 0 && SC(cpl) != 0)
        {
            fprintf(stderr, "Command failed with status code type %x and status code %x\n", SCT(cpl), SC(cpl));
            return EIO;
        }

        if (*CPL_CID(cpl) == expected_id)
        {
            uint16_t max_queues = _MIN((cpl->dword[0] >> 16) + 1, (cpl->dword[0] & 0xffff) + 1);
            controller->max_queues = max_queues;
        }
    }

    cq_update(&controller->admin_cq);

    // Extract information from identify structure
    unsigned char* bytes = controller->identify.virt_addr;
    controller->max_data_size = bytes[77] * (1 << (12 + CAP$MPSMIN(controller->reg_ptr)));
    controller->sq_entry_size = 1 << _RB(bytes[512], 3, 0);
    controller->cq_entry_size = 1 << _RB(bytes[513], 3, 0);
    controller->max_out_cmds = *((uint16_t*) (bytes + 514));
    controller->n_ns = *((uint32_t*) (bytes + 516));

    return 0;
}


static int set_num_queues(nvm_ctrl_t* controller)
{
    struct command* cmd = sq_enqueue(&controller->admin_sq);
    if (cmd == NULL)
    {
        return ENOSPC;
    }

    cmd_header(cmd, ADMIN_SET_FEATURES, 0);

    cmd->dword[10] = (1 << 31) | 0x07;
    cmd->dword[11] = ((uint32_t) controller->max_queues << 16) | controller->max_queues;

    sq_submit(&controller->admin_sq);

    struct completion* cpl = cq_dequeue_block(&controller->admin_cq, controller->timeout);
    if (cpl == NULL)
    {
        return ETIME;
    }
    sq_update(&controller->admin_sq, cpl);
    cq_update(&controller->admin_cq);

    return 0;
}


/* Helper function to clear state in a queue handle */
static void clear_queue(nvm_queue_t* queue, nvm_ctrl_t* ctrl, uint16_t no, int is_sq, volatile void* reg_ptr)
{
    queue->no = no;
    queue->max_entries = 0;
    queue->entry_size = is_sq ? ctrl->sq_entry_size : ctrl->cq_entry_size;
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 1;
    queue->virt_addr = NULL;
    queue->bus_addr = 0;
    //queue->host_db = is_sq ? SQ_DBL(ctrl->reg_ptr, queue->no, ctrl->dstrd) : CQ_DBL(ctrl->reg_ptr, queue->no, ctrl->dstrd);
    queue->db = is_sq ? SQ_DBL(reg_ptr, queue->no, ctrl->dstrd) : CQ_DBL(reg_ptr, queue->no, ctrl->dstrd);
}


/* Helper function to clear queue's page struct */
static void clear_page(page_t* page)
{
    page->id = -1;
    page->device_handle = 0;
    page->device = -1;
    page->virt_addr = NULL;
    page->page_size = 0;
    page->bus_addr = 0;
}


int nvm_init(nvm_ctrl_t* ctrl, sci_device_t device, volatile void* register_ptr)
{
    int err;

    // Read out controller capabilities of interest
    long page_size = sysconf(_SC_PAGESIZE);

    if (page_size == -1)
    {
        fprintf(stderr, "Failed to retrieve host page size: %s\n", strerror(errno));
        return errno;
    }

    uint8_t host_page_size = encode_page_size(page_size);
    uint8_t max_page_size = CAP$MPSMAX(register_ptr);
    uint8_t min_page_size = CAP$MPSMIN(register_ptr);

    if (!(min_page_size <= host_page_size && host_page_size <= max_page_size))
    {
        fprintf(stderr, "Host is configured with an unsupported page size\n");
        return ENOSPC;
    }

    // Set controller properties
    ctrl->device = device;
    ctrl->page_size = page_size;
    ctrl->dstrd = CAP$DSTRD(register_ptr);
    ctrl->timeout = CAP$TO(register_ptr) * 500UL;
    ctrl->max_queues = 0;
    ctrl->max_out_cmds = 0;
    ctrl->max_data_size = 0;
    ctrl->max_entries = CAP$MQES(register_ptr) + 1;   // CAP.MQES is a 0's based value
    ctrl->cq_entry_size = sizeof(struct completion);
    ctrl->sq_entry_size = sizeof(struct command);
    ctrl->reg_ptr = register_ptr;
    ctrl->n_ns = 0;

    clear_queue(&ctrl->admin_cq, ctrl, 0, 0, ctrl->reg_ptr);
    clear_page(&ctrl->admin_cq_page);
    clear_queue(&ctrl->admin_sq, ctrl, 0, 1, ctrl->reg_ptr);
    clear_page(&ctrl->admin_sq_page);
    clear_page(&ctrl->identify);

    // Create admin submission/completion queue pair
    err = get_page(-1, 10, &ctrl->admin_cq_page, device);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate queue memory: %s\n", strerror(err));
        nvm_free(ctrl);
        return err;
    }

    err = get_page(-1, 11, &ctrl->admin_sq_page, device);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate queue memory: %s\n", strerror(err));
        nvm_free(ctrl);
        return err;
    }

    // Allocate buffer for controller data
    err = get_page(-1, 12, &ctrl->identify, device);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate controller identify memory: %s\n", strerror(err));
        nvm_free(ctrl);
        return err;
    }

    // Reset controller
    fprintf(stderr, "Resetting controller...\n");
    reset_controller(ctrl);

    // Set admin CQ and SQ
    configure_admin_queues(ctrl);

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, ctrl->timeout);
    fprintf(stderr, "Controller online\n");

    // Submit identify controller command
    err = identify_controller(ctrl);
    if (err != 0)
    {
        fprintf(stderr, "Failed to identify controller: %s\n", strerror(err));
        nvm_free(ctrl);
        return err;
    }

    // Set CQES and SQES in CC
    configure_entry_sizes(ctrl);

    // Set features
    err = set_num_queues(ctrl);
    if (err != 0)
    {
        fprintf(stderr, "Failed to submit command: %s\n", strerror(err));
        nvm_free(ctrl);
        return err;
    }

    return 0;
}


void nvm_free(nvm_ctrl_t* ctrl)
{
    if (ctrl != NULL)
    {
        // TODO: send abort and prepare reset commands
        put_page(&ctrl->identify);
        put_page(&ctrl->admin_cq_page);
        put_page(&ctrl->admin_sq_page);
    }
}


int nvm_create_cq(nvm_ctrl_t* ctrl, nvm_queue_t* queue, uint16_t no, void* vaddr, uint64_t paddr, volatile void* regptr)
{
    if (!(0 < no && no <= ctrl->max_queues))
    {
        return EINVAL;
    }

    struct command* cmd = sq_enqueue(&ctrl->admin_sq);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    clear_queue(queue, ctrl, no, 0, regptr);
    queue->virt_addr = vaddr;
    queue->bus_addr = paddr;

    // Queues can never exceed one controller memory page size
    queue->max_entries = _MIN(ctrl->max_entries, ctrl->page_size / queue->entry_size);

    // Create completion queue command
    cmd_header(cmd, ADMIN_CREATE_COMPLETION_QUEUE, 0);
    cmd_data_ptr(cmd, queue->bus_addr, 0);

    cmd->dword[10] = (((uint32_t) queue->max_entries - 1) << 16) | queue->no;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;

    sq_submit(&ctrl->admin_sq);

    struct completion* cpl = cq_dequeue_block(&ctrl->admin_cq, ctrl->timeout);
    if (cpl == NULL)
    {
        return ETIME;
    }

    uint8_t sct = SCT(cpl);
    uint8_t sc = SC(cpl);

    sq_update(&ctrl->admin_sq, cpl);
    cq_update(&ctrl->admin_cq);

    if (sct != 0 && sc != 0)
    {
        return -1; // FIXME: look up error code in specification
    }

    return 0;
}


int nvm_create_sq(nvm_ctrl_t* ctrl, const nvm_queue_t* cq, nvm_queue_t* queue, uint16_t no, void* vaddr, uint64_t paddr, volatile void* regptr)
{
    if (!(0 < no && no <= ctrl->max_queues))
    {
        return EINVAL;
    }

    struct command* cmd = sq_enqueue(&ctrl->admin_sq);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    clear_queue(queue, ctrl, no, 1, regptr);
    queue->virt_addr = vaddr;
    queue->bus_addr = paddr;

    // Queues can never exceed one controller memory page size
    queue->max_entries = _MIN(ctrl->max_entries, ctrl->page_size / queue->entry_size);

    // Create completion queue command
    cmd_header(cmd, ADMIN_CREATE_SUBMISSION_QUEUE, 0);
    cmd_data_ptr(cmd, queue->bus_addr, 0);

    cmd->dword[10] = (((uint32_t) queue->max_entries - 1) << 16) | queue->no;
    cmd->dword[11] = (((uint32_t) cq->no) << 16) | (0x00 << 1) | 0x01;

    sq_submit(&ctrl->admin_sq);

    struct completion* cpl = cq_dequeue_block(&ctrl->admin_cq, ctrl->timeout);
    if (cpl == NULL)
    {
        return ETIME;
    }

    uint8_t sct = SCT(cpl);
    uint8_t sc = SC(cpl);

    sq_update(&ctrl->admin_sq, cpl);
    cq_update(&ctrl->admin_cq);

    if (sct != 0 && sc != 0)
    {
        return -1; // FIXME: look up error code in specification
    }

    return 0;
}

