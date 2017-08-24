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
 * Get error string.
 */
const char* nvm_strerror(const struct nvm_completion* cpl);


/*
 * Set command's DWORD0 and DWORD1
 */
void nvm_cmd_header(struct nvm_command* cmd, uint8_t opcode, uint32_t ns_id);


/*
 * Set command's DPTR field (DWORD6-9)
 */
void nvm_cmd_data_ptr(struct nvm_command* cmd, uint64_t prp1, uint64_t prp2);


/* Build a PRP list consisting of PRP entries 
 *
 * prp_list     - pointer to contiguous virtual memory where the PRP lists reside
 * page_size    - the controllers memory page size (MPS)
 * list_addrs   - array of bus addresses to prp_list's physical MPS-sized pages
 * prp_addrs    - array of n_prps elements containing bus address to describe PRPs
 *
 * Returns the bus address of the start of the PRP list
 */
uint64_t nvm_prp_list(void* prp_list, size_t page_size, size_t n_prps, const uint64_t* list_addrs, const uint64_t* prp_addrs);


#ifdef __cplusplus
}
#endif
#endif
