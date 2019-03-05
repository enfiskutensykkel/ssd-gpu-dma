#ifndef __NVM_INTERNAL_MUTEX_H__
#define __NVM_INTERNAL_MUTEX_H__

/* Forward declaration */
struct mutex;


/*
 * We currently only support OSes that have pthreads
 */
#if defined( __unix__ )
#include <pthread.h>
#else
#error "OS is not supported"
#endif



/*
 * We don't want another level of indirection by
 * hiding implementation and using pointers, so
 * we expose the struct definition here.
 */
#if defined( __unix__ )
struct mutex
{
    pthread_mutex_t mutex;
};
#endif



/*
 * Initialize mutex handle.
 */
int _nvm_mutex_init(struct mutex* mtx);



/*
 * Destroy mutex handle.
 */
int _nvm_mutex_free(struct mutex* mtx);



/*
 * Enter critical section.
 */
int _nvm_mutex_lock(struct mutex* mtx);



/*
 * Leave critical section.
 */
void _nvm_mutex_unlock(struct mutex* mtx);



#endif /* __NVM_INTERNAL_MUTEX_H__ */
