#include <nvm_types.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_rpc.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_ctrl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "admin.h"
#include "rpc.h"
#include "regs.h"
#include "util.h"
#include "dprintf.h"



void _nvm_admin_cq_create(nvm_cmd_t* cmd, const nvm_queue_t* cq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_CREATE_COMPLETION_QUEUE, 0);
    nvm_cmd_data_ptr(cmd, cq->ioaddr, 0);

    cmd->dword[10] = (((uint32_t) cq->max_entries - 1) << 16) | cq->no;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;
}



void _nvm_admin_cq_delete(nvm_cmd_t* cmd, const nvm_queue_t* cq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_DELETE_COMPLETION_QUEUE, 0);
    cmd->dword[10] = cq->no & 0xffff;
}



void _nvm_admin_sq_create(nvm_cmd_t* cmd, const nvm_queue_t* sq, const nvm_queue_t* cq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_CREATE_SUBMISSION_QUEUE, 0);
    nvm_cmd_data_ptr(cmd, sq->ioaddr, 0);

    cmd->dword[10] = (((uint32_t) sq->max_entries - 1) << 16) | sq->no;
    cmd->dword[11] = (((uint32_t) cq->no) << 16) | (0x00 << 1) | 0x01;
}



void _nvm_admin_sq_delete(nvm_cmd_t* cmd, const nvm_queue_t* sq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_DELETE_SUBMISSION_QUEUE, 0);
    cmd->dword[10] = sq->no & 0xffff;
}



void _nvm_admin_current_num_queues(nvm_cmd_t* cmd, bool set, uint16_t n_cqs, uint16_t n_sqs)
{
    nvm_cmd_header(cmd, 0, set ? NVM_ADMIN_SET_FEATURES : NVM_ADMIN_GET_FEATURES, 0);
    nvm_cmd_data_ptr(cmd, 0, 0);

    cmd->dword[10] = (0x00 << 8) | 0x07;
    cmd->dword[11] = set ? ((n_cqs - 1) << 16) | (n_sqs - 1) : 0;
}



void _nvm_admin_identify_ctrl(nvm_cmd_t* cmd, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_IDENTIFY, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x01;
    cmd->dword[11] = 0;
}



void _nvm_admin_identify_ns(nvm_cmd_t* cmd, uint32_t ns_id, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_IDENTIFY, ns_id);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x00;
    cmd->dword[11] = 0;
}



void _nvm_admin_get_log_page(nvm_cmd_t* cmd, uint32_t ns_id, uint64_t ioaddr, uint8_t log_id, uint64_t log_offset)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_GET_LOG_PAGE, ns_id);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (1024 << 16) | log_id;
    cmd->dword[11] = 0;
    cmd->dword[12] = (uint32_t)log_offset;
    cmd->dword[13] = (uint32_t)(log_offset >> 32);
}



int nvm_admin_ctrl_info(nvm_aq_ref ref, struct nvm_ctrl_info* info, void* ptr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (info == NULL || ptr == NULL || ioaddr == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(info, 0, sizeof(struct nvm_ctrl_info));
    memset(ptr, 0, 0x1000);

    nvm_cache_invalidate(ptr, 0x1000);

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    info->nvme_version = (uint32_t) *VER(ctrl->mm_ptr);
    info->page_size = ctrl->page_size;
    info->db_stride = 1UL << ctrl->dstrd;
    info->timeout = ctrl->timeout;
    info->contiguous = !!CAP$CQR(ctrl->mm_ptr);
    info->max_entries = ctrl->max_entries;

    _nvm_admin_identify_ctrl(&command, ioaddr);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Identify controller failed: %s\n", nvm_strerror(err));
        return err;
    }

    nvm_cache_invalidate(ptr, 0x1000);

    const unsigned char* bytes = (const unsigned char*) ptr;
    memcpy(info->pci_vendor, bytes, 4);
    memcpy(info->serial_no, bytes + 4, 20);
    memcpy(info->model_no, bytes + 24, 40);
    memcpy(info->firmware, bytes + 64, 8);

    info->max_data_size = (1UL << bytes[77]) * (1UL << (12 + CAP$MPSMIN(ctrl->mm_ptr)));
    info->max_data_pages = info->max_data_size / info->page_size;
    info->sq_entry_size = 1 << _RB(bytes[512], 3, 0);
    info->cq_entry_size = 1 << _RB(bytes[513], 3, 0);
    info->max_out_cmds = *((uint16_t*) (bytes + 514));
    info->max_n_ns = *((uint32_t*) (bytes + 516));

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_ns_info(nvm_aq_ref ref, struct nvm_ns_info* info, uint32_t ns_id, void* ptr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (info == NULL || ptr == NULL || ioaddr == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(ptr, 0, 0x1000);
    memset(info, 0, sizeof(struct nvm_ns_info));

    nvm_cache_invalidate(ptr, 0x1000);

    info->ns_id = ns_id;

    _nvm_admin_identify_ns(&command, ns_id, ioaddr);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Identify namespace failed: %s\n", nvm_strerror(err));
        return err;
    }
    
    nvm_cache_invalidate(ptr, 0x1000);

    const unsigned char* bytes = (const unsigned char*) ptr;
    info->size = *((uint64_t*) ptr);
    info->capacity = *((uint64_t*) (ptr + 8));
    info->utilization = *((uint64_t*) (ptr + 16));

    uint8_t format_idx = _RB(bytes[26], 3, 0);

    uint32_t lba_format = *((uint32_t*) (bytes + 128 + sizeof(uint32_t) * format_idx));
    info->lba_data_size = 1 << _RB(lba_format, 23, 16);
    info->metadata_size = _RB(lba_format, 15, 0);

    return NVM_ERR_PACK(NULL, 0);
}

int nvm_admin_get_log_page(nvm_aq_ref ref, uint32_t ns_id, void* ptr, uint64_t ioaddr, uint8_t log_id, uint64_t log_offset)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(ptr, 0, 0x1000);

    nvm_cache_invalidate(ptr, 0x1000);

    _nvm_admin_get_log_page(&command, ns_id, ioaddr, log_id, log_offset);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Get log page failed: %s\n", nvm_strerror(err));
        return err;
    }
    
    nvm_cache_invalidate(ptr, 0x1000);

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_cq_create(nvm_aq_ref ref, nvm_queue_t* cq, uint16_t id, void* vaddr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t queue;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    if (id == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    nvm_queue_clear(&queue, ctrl, true, id, vaddr, ioaddr);

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    _nvm_admin_cq_create(&command, &queue);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Creating completion queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    *cq = queue;
    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_cq_delete(nvm_aq_ref ref, nvm_queue_t* cq)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (cq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    _nvm_admin_cq_delete(&command, cq);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Deleting completion queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    cq->db = NULL;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, void* vaddr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t queue;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    if (id == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    nvm_queue_clear(&queue, ctrl, false, id, vaddr, ioaddr);

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    _nvm_admin_sq_create(&command, &queue, cq);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Creating submission queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    *sq = queue;
    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_sq_delete(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (cq == NULL || cq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    if (sq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    _nvm_admin_sq_delete(&command, sq);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Deleting submission queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    sq->db = NULL;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_get_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));

    _nvm_admin_current_num_queues(&command, false, 0, 0);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Failed to get current number of queues: %s\n", nvm_strerror(err));
        return err;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_set_num_queues(nvm_aq_ref ref, uint16_t n_cqs, uint16_t n_sqs)
{
    return nvm_admin_request_num_queues(ref, &n_cqs, &n_sqs);
}



int nvm_admin_request_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (*n_cqs == 0 || *n_sqs == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));

    _nvm_admin_current_num_queues(&command, true, *n_cqs, *n_sqs);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (err != 0)
    {
        dprintf("Failed to set current number of queues: %s\n", nvm_strerror(err));
        return err;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;

    return NVM_ERR_PACK(NULL, 0);
}

