#ifndef __DIS_NVM_UTIL_H__
#define __DIS_NVM_UTIL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
struct nvm_completion;

/* Convenience function for creating a bit mask */
static inline uint64_t _nvm_bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Get minimum of two values */
#define _MIN(a, b) ( (a) <= (b) ? (a) : (b) )

/* Get the maximum of two values */
#define _MAX(a, b) ( (a) > (b) ? (a) : (b) )


/* Extract specific bits */
#define _RB(v, hi, lo)   \
    ( ( (v) & _nvm_bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)   \
    ( ( (v) << (lo) ) & _nvm_bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))


#define DMA_ALIGN(vaddr, page_size) \
    ((vaddr) & ~((page_size) - 1))

#define DMA_MASK(vaddr, page_size) DMA_ALIGN((vaddr), (page_size))

#define DMA_SIZE(size, page_size) \
    (((size) + (page_size) - 1) & ~((page_size) - 1))

#define DMA_VADDR(vaddr, page_size, page) \
    ((void*) (((unsigned char*) (vaddr)) + ((page_size) * (page))))

#define DMA_WND_VADDR(window, page) \
    DMA_VADDR((window)->vaddr, (window)->page_size, (page))


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

#define CPL_OK(p)                   ( !SCT(p) && !SC(p) )

// #define NVM_ERR_PACK(cpl)

/* 
 * Get error string.
 */
const char* nvm_status(const struct nvm_completion* cpl);

#define nvm_strerror(cpl) nvm_status(cpl)

// TODO: make RPC functions pack nvm error into int32_t
//const char* nvm_strerror(int32_t error);

#ifdef __cplusplus
}
#endif
#endif
