#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/vfio.h>
#include "common.h"


/*
 * Bus-device-function descriptor.
 * Used to identify a device/function in the PCI tree.
 */
struct bdf
{
    int     domain;
    int     bus;
    int     device;
    int     function;
    int     fd;
    int     vfio_group;
    int     vfio_cfd;
};

static int pci_ioaddrs_iommu(int vfio_cfd, void* ptr, size_t page_size,
			     size_t n_pages, uint64_t* ioaddrs) {
    for (size_t i_page = 0; i_page < n_pages; ++i_page)
    {
	struct vfio_iommu_type1_dma_map dma_map = { .argsz = sizeof(dma_map) };
	dma_map.vaddr = (uint64_t)ptr + (i_page * page_size);
	dma_map.size = page_size;
	dma_map.iova = ioaddrs[i_page];
	dma_map.flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;

	int rc = ioctl(vfio_cfd, VFIO_IOMMU_MAP_DMA, &dma_map);
	if (rc != 0)
	  return -errno;
    }
    return 0;
}

static int lookup_ioaddrs(void* ptr, size_t page_size, size_t n_pages, uint64_t* ioaddrs)
{
    uint64_t vaddr = (uint64_t) ptr;
    size_t offset = (vaddr / page_size) * sizeof(void*);

    FILE* fp = fopen("/proc/self/pagemap", "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open page map: %s\n", strerror(errno));
        return errno;
    }

    if (fseek(fp, offset, SEEK_SET) != 0)
    {
        fclose(fp);
        fprintf(stderr, "Failed to seek: %s\n", strerror(errno));
        return errno;
    }

    if (fread(ioaddrs, sizeof(uint64_t), n_pages, fp) != n_pages)
    {
        fclose(fp);
        fprintf(stderr, "Failed to read: %s\n", strerror(errno));
        return errno;
    }

    fclose(fp);

    for (size_t i_page = 0; i_page < n_pages; ++i_page)
    {
        if (!(ioaddrs[i_page] & (1ULL << 63)))
        {
            fprintf(stderr, "Page not present in memory!\n");
            return EINVAL;
        }

        ioaddrs[i_page] = (ioaddrs[i_page] & ((1ULL << 54) - 1)) * page_size;
    }

    return 0;
}

static int identify(const nvm_ctrl_t* ctrl, uint32_t nvm_ns_id, int vfio_cfd)
{
    int status;
    void* memory;
    nvm_dma_t* window = NULL;
    nvm_aq_ref admin = NULL;
    uint64_t ioaddrs[3];

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
        fprintf(stderr, "Failed to look up page size: %s\n", strerror(errno));
        return 1;
    }

    status = posix_memalign(&memory, ctrl->page_size, 3 * page_size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory: %s\n", strerror(status));
        return 1;
    }

    status = mlock(memory, 3 * page_size);
    if (status != 0)
    {
        free(memory);
        fprintf(stderr, "Failed to page-lock memory: %s\n", strerror(status));
        return 1;
    }

    status = lookup_ioaddrs(memory, page_size, 3, ioaddrs);
    if (status != 0)
    {
        munlock(memory, 3 * page_size);
        free(memory);
        goto out;
    }

    if (vfio_cfd >= 0) {
	status = pci_ioaddrs_iommu(vfio_cfd, memory, page_size, 3, ioaddrs);
	if (status != 0)
	{
	    munlock(memory, 3 * page_size);
	    free(memory);
	    goto out;
	}
    }

    status = nvm_dma_map(&window, ctrl, memory, page_size, 3, ioaddrs);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));
        status = 2;
        goto out;
    }

    admin = reset_ctrl(ctrl, window);
    if (admin == NULL)
    {
        goto out;
    }

    status = identify_ctrl(admin, ((unsigned char*) memory) + 2 * ctrl->page_size, ioaddrs[2]);
    if (status != 0)
    {
        goto out;
    }

    if (nvm_ns_id != 0)
    {
        status = identify_ns(admin, nvm_ns_id, ((unsigned char*) memory) + 2 * ctrl->page_size, ioaddrs[2]);
    }

out:
    nvm_aq_destroy(admin);
    nvm_dma_unmap(window);
    munlock(memory, 3 * page_size);
    free(memory);
    return status;
}


static int pci_enable_device(const struct bdf* dev)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/enable",
            dev->domain, dev->bus, dev->device, dev->function);

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open file descriptor: %s\n", strerror(errno));
        return errno;
    }

    fputc('1', fp);
    fclose(fp);
    return 0;
}


/*
 * Allow device to do DMA.
 */
static int pci_set_bus_master(const struct bdf* dev)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/config", 
            dev->domain, dev->bus, dev->device, dev->function);

    FILE* fp = fopen(path, "r+");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open config space file: %s\n", strerror(errno));
        return errno;
    }

    uint16_t command;
    fseek(fp, 0x04, SEEK_SET);
    fread(&command, sizeof(command), 1, fp);

    if ((command & (1 << 0x02)) == 0) {
	command |= (1 << 0x02);

	fseek(fp, 0x04, SEEK_SET);
	fwrite(&command, sizeof(command), 1, fp);
    }

    fclose(fp);
    return 0;
}


/*
 * Open a file descriptor to device memory.
 */
static int pci_open_bar(struct bdf* dev, int bar)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/resource%d", 
            dev->domain, dev->bus, dev->device, dev->function, bar);

    dev->fd = open(path, O_RDWR);
    if (dev->fd < 0)
    {
        fprintf(stderr, "Failed to open resource file: %s\n", strerror(errno));
	return -1;
    }
    return 0;
}

static int pci_open_vfio(struct bdf* dev, int bar, volatile void** ctrl_registers)
{
    char path[64];
    int group_fd, rc;
    struct vfio_group_status group_status;
    struct vfio_region_info region_info = { .argsz = sizeof(region_info),
                                            .index = VFIO_PCI_BAR0_REGION_INDEX};

    if (bar != 0) {
	fprintf(stderr, "Only support mapping BAR 0\n");
	return -1;
    }

    /* Create a new container */
    dev->vfio_cfd = open("/dev/vfio/vfio", O_RDWR);
    if (dev->vfio_cfd < 0) {
        fprintf(stderr, "VFIO: Error opening /dev/vfio/vfio (%d)\n", errno);
	return -1;
    }

    if (ioctl(dev->vfio_cfd, VFIO_GET_API_VERSION) != VFIO_API_VERSION) {
        fprintf(stderr, "VFIO: Unknown API version\n");
	goto error_close_container_fd;
    }

    if (!ioctl(dev->vfio_cfd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU)) {
        fprintf(stderr, "VFIO: Doesn't support the VFIO_TYPE1_IOMMU driver (%d)\n", errno);
	goto error_close_container_fd;
    }

    /* Open the group */
    sprintf(path, "/dev/vfio/%d", dev->vfio_group);
    group_fd = open(path, O_RDWR);
    if (group_fd < 0) {
        fprintf(stderr, "VFIO: Error opening %s (%d)\n", path, errno);
	goto error_close_container_fd;
    }

    /* Test the group is viable and available */
    bzero(&group_status, sizeof(group_status));
    group_status.argsz = sizeof(group_status);
    rc = ioctl(group_fd, VFIO_GROUP_GET_STATUS, &group_status);
    if (rc != 0) {
        fprintf(stderr, "VFIO: Failed to get group status rc=%d, errno=%d\n", rc, errno);
	goto error_close_group_fd;
    }
    if (!(group_status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
        fprintf(stderr, "VFIO: Group is not viable (ie, not all devices bound for vfio)\n");
	goto error_close_group_fd;
    }

    /* Add the group to the container */
    rc = ioctl(group_fd, VFIO_GROUP_SET_CONTAINER, &dev->vfio_cfd);
    if (rc != 0) {
        fprintf(stderr, "VFIO: Failed to set group to container (%d)\n", errno);
	goto error_close_group_fd;
    }

    /* Enable the IOMMU model we want */
    if (ioctl(dev->vfio_cfd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) != 0) {
        fprintf(stderr, "VFIO: Failed to set IOMMU model (%d)\n", errno);
	goto error_close_group_fd;
    }

    /* Get a file descriptor for the device */
    sprintf(path, "%04x:%02x:%02x.%x",
            dev->domain, dev->bus, dev->device, dev->function);
    dev->fd = ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, path);
    if (dev->fd < 0) {
        fprintf(stderr, "VFIO: Error opening device %s (%d)\n", path, errno);
	goto error_close_group_fd;
    }

    /* Test and setup the device */
    rc = ioctl(dev->fd, VFIO_DEVICE_GET_REGION_INFO, &region_info);
    if (rc < 0) {
        fprintf(stderr, "VFIO: Error get device %s info (%d)\n", path, errno);
	goto error_close_dev_fd;
    }

    __u32 required_flags = VFIO_REGION_INFO_FLAG_READ |
			   VFIO_REGION_INFO_FLAG_WRITE |
			   VFIO_REGION_INFO_FLAG_MMAP;
    if ((region_info.flags & required_flags) != required_flags) {
        fprintf(stderr, "VFIO: Device %s does not have required flags\n", path);
	goto error_close_dev_fd;
    }

    if (region_info.size < NVM_CTRL_MEM_MINSIZE) {
        fprintf(stderr, "VFIO: Device %s region too small %llu < %u\n",
		path, region_info.size, NVM_CTRL_MEM_MINSIZE);
	goto error_close_dev_fd;
    }

    *ctrl_registers = mmap(NULL, NVM_CTRL_MEM_MINSIZE,
	                   PROT_READ | PROT_WRITE,
			   MAP_SHARED, dev->fd, region_info.offset);
    if (*ctrl_registers == NULL || *ctrl_registers == MAP_FAILED)
    {
	fprintf(stderr, "Failed to memory map BAR reasource file: %s\n", strerror(errno));
	goto error_close_dev_fd;
    }

    /* Gratuitous device reset and go... */
    rc = ioctl(dev->fd, VFIO_DEVICE_RESET);
    if (rc < 0) {
        fprintf(stderr, "VFIO: Error reset device %s (%d)\n", path, errno);
	goto error_close_dev_fd;
    }

    close(group_fd);
    return 0;

error_close_dev_fd:
    close(dev->fd);
    dev->fd = -1;
error_close_group_fd:
    close(group_fd);
error_close_container_fd:
    close(dev->vfio_cfd);
    dev->vfio_cfd = -1;
    return -1;
}

static void parse_args(int argc, char** argv, struct bdf* device, uint32_t* nvm_ns_id);


int main(int argc, char** argv)
{
    int status;
    nvm_ctrl_t* ctrl;

    uint32_t nvm_ns_id;
    struct bdf device;
    volatile void* ctrl_registers = NULL;
    parse_args(argc, argv, &device, &nvm_ns_id);

    // Enable device
    status = pci_enable_device(&device);
    if (status != 0)
    {
        fprintf(stderr, "Failed to enable device %04x:%02x:%02x.%x\n",
                device.domain, device.bus, device.device, device.function);
        exit(1);
    }

    // Enable device DMA
    status = pci_set_bus_master(&device);
    if (status != 0)
    {
        fprintf(stderr, "Failed to access device config space %04x:%02x:%02x.%x\n",
                device.domain, device.bus, device.device, device.function);
        exit(2);
    }

    if (device.vfio_group >= 0) {
	status = pci_open_vfio(&device, 0, &ctrl_registers);
	if (status < 0) {
	    fprintf(stderr, "Failed to setup PCI mapping\n");
	    exit(3);
	}
    } else {
	// Memory-map device memory
	status = pci_open_bar(&device, 0);
	if (status < 0)
	{
	    fprintf(stderr, "Failed to access device BAR memory\n");
	    exit(3);
	}

	ctrl_registers = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, device.fd, 0);
	if (ctrl_registers == NULL || ctrl_registers == MAP_FAILED)
	{
	    fprintf(stderr, "Failed to memory map BAR reasource file: %s\n", strerror(errno));
	    close(device.fd);
	    exit(3);
	}
    }

    // Get controller reference
    status = nvm_raw_ctrl_init(&ctrl, ctrl_registers, NVM_CTRL_MEM_MINSIZE);
    if (status != 0)
    {
        munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
        close(device.fd);
	if (device.vfio_cfd >= 0)
	    close(device.vfio_cfd);
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(4);
    }

    status = identify(ctrl, nvm_ns_id, device.vfio_cfd);

    nvm_ctrl_free(ctrl);
    munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
    close(device.fd);
    if (device.vfio_cfd >= 0)
	close(device.vfio_cfd);

    fprintf(stderr, "Goodbye!\n");
    exit(status);
}


static void give_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl=<pci bdf>\n", name);
}


static void show_help(const char* name)
{
    give_usage(name);
    fprintf(stderr, "\nCreate a manager and run an IDENTIFY CONTROLLER NVM admin command.\n\n"
            "    --ctrl     <pci bdf>       PCI bus-device-function to controller.\n"
            "    --ns       <namespace>     Show information about NVM namespace.\n"
            "    --help                     Show this information.\n\n");
}


static int parse_bdf(char* str, struct bdf* dev)
{
    char* colon = strrchr(str, ':');
    char* dot = strchr(colon != NULL ? colon : str, '.');
    char* endptr;

    char* function = "0";
    char* slot = str;
    char* bus = "0";
    char* domain = "0";

    if (colon != NULL)
    {
        bus = str;
        *colon++ = '\0';
        slot = colon;

        colon = strchr(str, ':');
        if (colon != NULL)
        {
            domain = str;
            *colon++ = '\0';
            bus = colon;
        }
    }

    if (dot != NULL)
    {
        *dot++ = '\0';
        function = dot;
    }

    dev->domain = strtoul(domain, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->domain > 0xffff)
    {
        fprintf(stderr, "Invalid PCI domain number: '%s'\n", domain);
        return 1;
    }

    dev->bus = strtoul(bus, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->bus > 0xff)
    {
        fprintf(stderr, "Invalid PCI bus number: '%s'\n", bus);
        return 1;
    }

    dev->device = strtoul(slot, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->device > 0xff)
    {
        fprintf(stderr, "Invalid PCI device number: '%s'\n", slot);
        return 1;
    }

    dev->function = strtoul(function, &endptr, 0);
    if (endptr == NULL || *endptr != '\0')
    {
        fprintf(stderr, "Invalid PCI device function: '%s'\n", function);
        return 1;
    }

    return 0;
}



static void parse_args(int argc, char** argv, struct bdf* dev, uint32_t* nvm_ns_id)
{
    // Command line options
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "ns", required_argument, NULL, 'n' },
        { "vfio", required_argument, NULL, 'v' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;
    char* endptr = NULL;
    bool dev_set = false;
    
    *nvm_ns_id = 0;
    memset(dev, 0, sizeof(struct bdf));
    dev->vfio_group = -1;
    dev->vfio_cfd   = -1;

    while ((opt = getopt_long(argc, argv, ":hc:n:", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?': // unknown option
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                exit('?');

            case ':': // missing option argument
                fprintf(stderr, "Missing argument for option: `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                exit(':');

            case 'c': // device identifier
                if (parse_bdf(optarg, dev) != 0)
                {
                    give_usage(argv[0]);
                    exit('c');
                }
                dev_set = true;
                break;

            case 'n': // namespace identifier
                *nvm_ns_id = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid NVM namespace!\n");
                    exit('n');
                }
                break;

            case 'v':
                dev->vfio_group = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid vfio group!\n");
                    exit('v');
                }
		break;

            case 'h':
                show_help(argv[0]);
                exit(0);
        }
    }

    // Check if controller is set
    if (!dev_set)
    {
        fprintf(stderr, "Controller is not specified!\n");
        exit('c');
    }

    // Check if device actually exists
    char path[64];
    struct stat s;
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x",
            dev->domain, dev->bus, dev->device, dev->function);

    if (stat(path, &s) != 0 || ! S_ISDIR(s.st_mode))
    {
        fprintf(stderr, "%04x:%02x:%02x.%x is not a valid PCI device\n",
                dev->domain, dev->bus, dev->device, dev->function);
        exit('c');
    }
}

