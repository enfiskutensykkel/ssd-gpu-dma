#ifndef __NVM_INTERNAL_UTIL_H__
#define __NVM_INTERNAL_UTIL_H__

#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
//#include <pthread.h>

#ifndef NDEBUG
#include <string.h>
#include <errno.h>
#include "dprintf.h"
#endif


/* Get minimum of two values */
#define _MIN(a, b) ( (a) <= (b) ? (a) : (b) )


/* Get the maximum of two values */
#define _MAX(a, b) ( (a) > (b) ? (a) : (b) )



/* Calculate the base-2 logarithm of a number n */
static inline uint32_t _nvm_b2log(uint32_t n)
{
    uint32_t count = 0;

    while (n > 0)
    {
        ++count;
        n >>= 1;
    }

    return count - 1;
}


/* Delay the minimum of one millisecond and a time remainder */
static inline uint64_t _nvm_delay_remain(uint64_t remaining_nanoseconds)
{
    struct timespec ts;

    if (remaining_nanoseconds == 0)
    {
        return 0;
    }

    ts.tv_sec = 0;
    ts.tv_nsec = _MIN(1000000UL, remaining_nanoseconds);

    //pthread_yield();
    clock_nanosleep(CLOCK_REALTIME, 0, &ts, NULL);

    remaining_nanoseconds -= _MIN(1000000UL, remaining_nanoseconds);
    return remaining_nanoseconds;
}


/* Get the system page size */
static inline size_t _nvm_host_page_size()
{
    long page_size = sysconf(_SC_PAGESIZE);

#ifndef NDEBUG
    if (page_size < 0)
    {
        dprintf("Failed to look up system page size: %s\n", strerror(errno));
        return 0;
    }
#endif

    return page_size;
}


#endif /* __NVM_INTERNAL_UTIL_H__ */
