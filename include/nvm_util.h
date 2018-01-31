#ifndef __NVM_UTIL_H__
#define __NVM_UTIL_H__

#include <nvm_types.h>
#include <stdint.h>

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

/* Convenience function for creating a bit mask */
static inline __device__ __host__
uint64_t _nvm_bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Extract specific bits */
#define _RB(v, hi, lo)      \
    ( ( (v) & _nvm_bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)      \
    ( ( (v) << (lo) ) & _nvm_bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))


/*
 * Calculate block number from page number.
 */
#define NVM_PAGE_TO_BLOCK(page_size, block_size, pageno)    \
    (((page_size) * (pageno)) / (block_size))
    


/*
 * Calculate page number from block number.
 */
#define NVM_BLOCK_TO_PAGE(page_size, block_size, blockno)   \
    (((block_size) * (blockno)) / (page_size))


/*
 * Create mask to clear away address offset.
 */
#define NVM_PAGE_MASK(page_size)                    \
    ~((page_size) - 1)


/*
 * Round address down to nearest page alignment.
 */
#define NVM_ADDR_MASK(addr, page_size)              \
    (((uint64_t) (addr)) & NVM_PAGE_MASK((page_size)))



/*
 * Align size to page boundary.
 */
#define NVM_PAGE_ALIGN(size, page_size)             \
    (((size) + (page_size) - 1) & NVM_PAGE_MASK((page_size)))


/*
 * Calculate page-aligned offset into address.
 */
#define NVM_ADDR_OFFSET(addr, page_size, pageno)    \
    (((uint64_t) (addr)) + ((page_size) * (pageno)))


/*
 * Calculate page-aligned offset into pointer.
 */
#define NVM_PTR_OFFSET(ptr, page_size, pageno)      \
    ((void*) (((unsigned char*) (ptr)) + ((page_size) * (pageno))))


/*
 * Align size to controller pages.
 */
#define NVM_CTRL_ALIGN(ctrl_ptr, size)              \
    NVM_PAGE_ALIGN((size), (ctrl_ptr)->page_size)


/*
 * Convert size to number of controller pages.
 */
#define NVM_CTRL_PAGES(ctrl_ptr, size)              \
    (NVM_CTRL_ALIGN((ctrl_ptr), (size)) / (ctrl_ptr)->page_size)


/*
 * Align size to page size.
 */
#define NVM_DMA_ALIGN(dma_ptr, size)                \
    NVM_PAGE_ALIGN((size), (dma_ptr)->page_size)


/*
 * Calculate controller page-aligned offset into DMA handle pointer.
 */
#define NVM_DMA_OFFSET(dma_ptr, pageno)             \
    NVM_PTR_OFFSET((dma_ptr)->vaddr, (dma_ptr)->page_size, (pageno))




/* Standard fields in a command */
#define NVM_CMD_CID(p)              _REG(p, 2, 16)
#define NVM_CMD_NSID(p)             _REG(p, 1, 32)


/* Standard fields in a completion */
#define NVM_CPL_CID(p)              _REG(p, 12, 16)
#define NVM_CPL_SQHD(p)             _REG(p,  8, 16)
#define NVM_CPL_SQID(p)             _REG(p, 10, 16)
#define NVM_CPL_STATUS(p)           _REG(p, 14, 16)



#ifdef __cplusplus
extern "C" {
#endif
/*
 * Get controller associated with admin queue-pair reference.
 */
const nvm_ctrl_t* nvm_ctrl_from_aq_ref(nvm_aq_ref ref);
#ifdef __cplusplus
}
#endif


#endif /* __NVM_UTIL_H__ */
