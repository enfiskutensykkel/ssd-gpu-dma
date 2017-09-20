#include <nvm_types.h>
#define __CUDA__
#include <nvm_ioctl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string.h>
#include <errno.h>


int main()
{
    int fd = open("/dev/disnvme0", O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open fd: %s\n", strerror(errno));
        return 1;
    }

    void* buffer;
    uint64_t ioaddrs[3];
    memset(ioaddrs, 0, sizeof(ioaddrs));

    posix_memalign(&buffer, 0x1000, 3 * 0x1000);
    printf("%p\n", buffer);

    struct nvm_ioctl_map map;
    map.vaddr_start = (uint64_t) buffer;
    map.n_pages = 3;
    map.ioaddrs = ioaddrs;

    int err = ioctl(fd, NVM_MAP_HOST_MEMORY, &map); 
    if (err != 0)
    {
        fprintf(stderr, "ioctl returned %s\n", strerror(errno));
    }

    for (int i = 0; i < 3; ++i)
    {
        printf("%lx\n", ioaddrs[i]);
    }

    return 0;
}
