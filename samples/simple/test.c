#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>


static int pci_set_bus_master(int domain, int bus, int dev, int fun)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/config", 
            domain, bus, dev, fun);

    FILE* fp = fopen(path, "r+");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open config space file: %s\n", strerror(errno));
        return errno;
    }

    uint16_t command;
    fseek(fp, 0x04, SEEK_SET);
    fread(&command, sizeof(command), 1, fp);

    command |= (1 << 0x02);

    fseek(fp, 0x04, SEEK_SET);
    fwrite(&command, sizeof(command), 1, fp);

    fclose(fp);
    return 0;
}


static int pci_open_bar(int domain, int bus, int dev, int fun, int bar)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/resource%d", 
            domain, bus, dev, fun, bar);

    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open resource file: %s\n", strerror(errno));
    }

    return fd;
}


static int lookup_bus_addrs(void* ptr, uint64_t* bus_addr, size_t page_size)
{
    uint64_t vaddr = (uint64_t) ptr;
    off_t offset = (vaddr / page_size) * sizeof(void*);

    FILE* fp = fopen("/proc/self/pagemap", "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open this process' pagemap\n");
        return errno;
    }

    if (fseek(fp, offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "Failed to seek in pagemap: %s\n", strerror(errno));
        return errno;
    }

    uint64_t entry;
    if (fread(&entry, sizeof(uint64_t), 1, fp) != 1)
    {
        fprintf(stderr, "Failed to read pagemap: %s\n", strerror(errno));
        return errno;
    }

    if (!(entry & (1ULL << 63)))
    {
        fprintf(stderr, "Page not present in memory\n");
        return EINVAL;
    }

    fclose(fp);

    *bus_addr = (entry & ((1ULL << 54) - 1)) * page_size;
    return 0;
}


int main()
{
    int err;

    int domain = 0;
    int bus = 5;
    int dev = 0;
    int fun = 0;

    size_t page_size = sysconf(_SC_PAGESIZE);

    err = pci_set_bus_master(domain, bus, dev, fun);
    if (err != 0)
    {
        fprintf(stderr, "Failed to access device config space: %04x:%02x:%02x.%x\n",
                domain, bus, fun, fun);
        return 2;
    }

    int reg_fd = pci_open_bar(domain, bus, dev, fun, 0);
    if (reg_fd < 0)
    {
        fprintf(stderr, "Failed to access device BAR\n");
        return 2;
    }
    
    volatile void* reg_ptr = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, reg_fd, 0);
    if (reg_ptr == NULL || reg_ptr == MAP_FAILED)
    {
        fprintf(stderr, "Failed to mmap device BAR resource file: %s\n", strerror(errno));
        goto close_fd;
    }

    nvm_ctrl_t ctrl;
    err = nvm_ctrl_init_raw(&ctrl, reg_ptr, NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        fprintf(stderr, "Failed to initialize controller: %s\n", strerror(err));
        goto unmap;
    }

    void* queue_ptr;
    err = posix_memalign(&queue_ptr, ctrl.page_size, 3 * ctrl.page_size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory for admin queues: %s\n", strerror(err));
        goto free_ctrl;
    }

    err = mlock(queue_ptr, 3 * ctrl.page_size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to page-lock queue memory: %s\n", strerror(err));
        goto free_queue_mem;
    }

    uint64_t bus_addrs[3];
    for (size_t i = 0; i < 3; ++ i)
    {
        void* ptr = ((unsigned char*) queue_ptr) + i * page_size;

        err = lookup_bus_addrs(ptr, &bus_addrs[i], page_size);
        if (err != 0)
        {
            fprintf(stderr, "Failed to look up bus addresses: %s\n", strerror(err));
            goto unlock;
        }

        fprintf(stderr, "Using physical address %lx for ptr=%p\n", bus_addrs[i], ptr);
    }

    nvm_mngr_t mngr;
    err = nvm_mngr_init_raw(&mngr, &ctrl, queue_ptr, 3, bus_addrs);
    if (err != 0)
    {
        fprintf(stderr, "Failed to initialize queue manager: %s\n", strerror(err));
        goto unlock;
    }

    fprintf(stderr, "Hello, world\n");

    nvm_ctrl_info_t info;
    err = nvm_ctrl_get_info(mngr, &info);
    if (err != 0)
    {
        fprintf(stderr, "Failed to identify controller: %s\n", strerror(err));
    }
    
    info.serial_no[19] = 0;
    info.model_no[39] = 0;
    printf("%s\n", info.serial_no);
    printf("%s\n", info.model_no);

    nvm_mngr_free(mngr);
    
unlock:
    munlock(queue_ptr, 3 * ctrl.page_size);
free_queue_mem:
    free(queue_ptr);
free_ctrl:
    nvm_ctrl_free(&ctrl);
unmap:
    munmap((void*) reg_ptr, NVM_CTRL_MEM_MINSIZE);
close_fd:
    close(reg_fd);

    return err;
}
