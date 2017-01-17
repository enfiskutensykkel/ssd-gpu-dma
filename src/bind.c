#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kallsyms.h>
#include "bind.h"


/* Kernel must be built using CONFIG_KALLSYMS in order to get hold of NVMe driver functions */
#ifndef CONFIG_KALLSYMS
#error "Kernel must be built using CONFIG_KALLSYMS"
#endif


func_t* bind_func(size_t idx, const char* name)
{
    unsigned long addr;
    func_t* func;

    addr = kallsyms_lookup_name(name);
    if (addr == 0)
    {
        printk(KERN_ERR "Could not resolve symbol: %s\n", name);
        return ERR_PTR(-ENOMEM);
    }

    func = &nvme_funcs[idx];

    func->addr = addr;
    func->owner = __module_text_address(addr);

    printk(KERN_DEBUG "Binding function '%s' at addr 0x%08lx to index %zu\n", name, addr, idx);
    return func;
}


void unbind_all(size_t num)
{
    size_t i;

    for (i = 0; i < num; ++i)
    {
        if (nvme_funcs[i].owner)
        {
            module_put(nvme_funcs[i].owner);
        }

        nvme_funcs[i].addr = 0;
        nvme_funcs[i].owner = NULL;
    }
}
