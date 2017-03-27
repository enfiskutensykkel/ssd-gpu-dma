#ifndef __NVME_COMMAND_H__
#define __NVME_COMMAND_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "util.h"
#include <stdint.h>
#include <stddef.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


/* Standard fields in a command */
#define CMD_CID(p)                  _REG(p, 2, 16)
#define CMD_NSID(p)                 _REG(p, 1, 32)


/* Standard fields in a completion */
#define CPL_CID(p)                  _REG(p, 12, 16)
#define CPL_SQHD(p)                 _REG(p,  8, 16)
#define CPL_SQID(p)                 _REG(p, 10, 16)
#define CPL_STATUS(p)               _REG(p, 14, 16)


/* Status information in a completion */
#define SCT(p)                      ((uint8_t) _RB((p)->dword[3], 27, 25))
#define SC(p)                       ((uint8_t) _RB((p)->dword[3], 24, 17))



/* NVM completion data type (16 bytes) */
struct completion
{
    uint32_t dword[4];
};


/* NVM command data type (64 bytes) */
struct command
{
    uint32_t dword[16];
};


/* List of NVM command opcodes */
enum nvm_command_set
{
    NVM_READ                        = (0x00 << 7) | (0x00 << 2) | 0x02,
    NVM_WRITE                       = (0x00 << 7) | (0x00 << 2) | 0x01
};


/* Build a PRP list consisting of PRP entries 
 *
 * page_size    - the controllers memory page size (MPS)
 * prp_list     - pointer to contiguous virtual memory where the PRP lists reside
 * list_addrs   - array of bus addresses to prp_list's physical MPS-sized pages
 * prp_addrs    - array of n_prps elements containing bus address to describe PRPs
 *
 * Returns the bus address of the start of the PRP list
 */
__host__ __device__
uint64_t build_prp_list(size_t page_size, void* prp_list, size_t n_prps, const uint64_t* list_addrs, const uint64_t* prp_addrs);



/* Set command's DWORD0 and DWORD1 */
__host__ __device__
void cmd_header(struct command* cmd, uint8_t opcode, uint32_t ns_id);


/* 
 * Set command's DPTR field (DWORD6-9)
 */
__host__ __device__
void cmd_data_ptr(struct command* cmd, uint64_t prp1, uint64_t prp2);


#ifdef __cplusplus
}
#endif
#endif
