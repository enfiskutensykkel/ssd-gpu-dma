#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_admin.h>
#include <nvm_command.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>


void nvm_admin_cq_create(struct nvm_command* cmd, const struct nvm_queue* cq)
{
    nvm_cmd_header(cmd, NVM_ADMIN_CREATE_COMPLETION_QUEUE, 0);
    nvm_cmd_data_ptr(cmd, cq->ioaddr, 0);

    cmd->dword[10] = (((uint32_t) cq->max_entries - 1) << 16) | cq->no;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;
}


void nvm_admin_sq_create(struct nvm_command* cmd, const struct nvm_queue* sq, const struct nvm_queue* cq)
{
    nvm_cmd_header(cmd, NVM_ADMIN_CREATE_SUBMISSION_QUEUE, 0);
    nvm_cmd_data_ptr(cmd, sq->ioaddr, 0);

    cmd->dword[10] = (((uint32_t) sq->max_entries - 1) << 16) | sq->no;
    cmd->dword[11] = (((uint32_t) cq->no) << 16) | (0x00 << 1) | 0x01;
}


void nvm_admin_current_num_queues(struct nvm_command* cmd, int set, uint16_t n_cqs, uint16_t n_sqs)
{
    nvm_cmd_header(cmd, !!set ? NVM_ADMIN_SET_FEATURES : NVM_ADMIN_GET_FEATURES, 0);
    nvm_cmd_data_ptr(cmd, 0, 0);

    cmd->dword[10] = (0x00 << 8) | 0x07;
    cmd->dword[11] = !!set ? ((n_cqs - 1) << 16) | (n_sqs - 1) : 0;
}


void nvm_admin_identify_ctrl(struct nvm_command* cmd, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, NVM_ADMIN_IDENTIFY, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x01;
    cmd->dword[11] = 0;
}


void nvm_admin_identify_ns(struct nvm_command* cmd, uint32_t ns_id, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, NVM_ADMIN_IDENTIFY, ns_id);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x00;
    cmd->dword[11] = 0;
}

