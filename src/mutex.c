#ifdef __unix__
#include <pthread.h>
#include <string.h>
#endif

#include "mutex.h"
#include "dprintf.h"



#ifdef __unix__
int _nvm_mutex_init(struct mutex* mtx)
{
    int err;

    err = pthread_mutex_init(&mtx->mutex, NULL);
    if (err != 0)
    {
        dprintf("Failed to initialize mutex: %s\n", strerror(err));
        return err;
    }

    return 0;
}
#endif



#ifdef __unix__
int _nvm_mutex_free(struct mutex* mtx)
{
    return pthread_mutex_destroy(&mtx->mutex);
}
#endif



#ifdef __unix__
int _nvm_mutex_lock(struct mutex* mtx)
{
    pthread_mutex_lock(&mtx->mutex);
    return 0;
}
#endif



#ifdef __unix__
void _nvm_mutex_unlock(struct mutex* mtx)
{
    pthread_mutex_unlock(&mtx->mutex);
}
#endif

