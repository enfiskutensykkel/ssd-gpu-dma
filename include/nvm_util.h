#ifndef __DIS_NVM_UTIL_H__
#define __DIS_NVM_UTIL_H__

#include <stdint.h>


/* Convenience function for creating a bit mask */
static inline uint64_t bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Convenience function for getting the base-2 logarithm of a number */
static inline uint32_t b2log(uint32_t n)
{
    uint32_t count = 0;

    while (n > 0)
    {
        ++count;
        n >>= 1;
    }

    return count - 1;
}


/* Get minimum of two values */
#define _MIN(a, b) ( (a) <= (b) ? (a) : (b) )


/* Extract specific bits */
#define _RB(v, hi, lo)   \
    ( ( (v) & bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)   \
    ( ( (v) << (lo) ) & bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))


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

#endif
