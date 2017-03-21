#ifndef __NVM_UTIL_H__
#define __NVM_UTIL_H__

#include <stdint.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


#define _MIN(a, b) ( (a) <= (b) ? (a) : (b) )

/* Convenience function for creating a bit mask */
__host__ __device__
inline uint64_t bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Extract specific bits */
#define _RB(v, hi, lo)   \
    ( ( (v) & bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)   \
    ( ( (v) << (lo) ) & bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))


#endif
