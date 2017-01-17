#ifndef __SSD_DMA_BIND_H__
#define __SSD_DMA_BIND_H__

#include <linux/types.h>

typedef struct {
    unsigned long   addr;   /* function address */
    struct module*  owner;  /* module owner */
} func_t;


extern func_t nvme_funcs[];


func_t* bind_func(size_t idx, const char* name);


void unbind_all(size_t num);

#endif
