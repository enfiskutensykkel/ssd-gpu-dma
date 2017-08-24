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

