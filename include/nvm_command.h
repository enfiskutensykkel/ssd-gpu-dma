#ifndef __DIS_NVM_COMMAND_H__
#define __DIS_NVM_COMMAND_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

struct nvm_command;
struct nvm_completion;


/* List of NVM IO command opcodes */
enum nvm_io_command_set
{
    NVM_IO_READ                     = (0x00 << 7) | (0x00 << 2) | 0x02,
    NVM_IO_WRITE                    = (0x00 << 7) | (0x00 << 2) | 0x01
};


/*
 * Set command's DWORD0 and DWORD1
 */
void nvm_cmd_header(struct nvm_command* cmd, uint8_t opcode, uint32_t ns_id);


/*
 * Set command's DPTR field (DWORD6-9)
 */
void nvm_cmd_data_ptr(struct nvm_command* cmd, uint64_t prp1, uint64_t prp2);
#define nvm_cmd_dptr(cmd, prp1, prp2) nvm_cmd_data_ptr((cmd), (prp1), (prp2))


/*
 * Calculate number of pages needed to build a PRP list.
 */
size_t nvm_num_prp_pages(size_t page_size, size_t transfer_size);


/* 
 * Build a PRP list consisting of PRP entries.
 */
size_t nvm_prp_list(void* list_vaddr, size_t page_size, size_t size, const uint64_t* list_ioaddrs, const uint64_t* data_ioaddrs);


#ifdef __cplusplus
}
#endif
#endif
