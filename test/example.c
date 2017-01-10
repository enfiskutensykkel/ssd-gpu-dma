//#define _GNU_SOURCE
#include <ssd_dma.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <stdbool.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <linux/fs.h>
#include <signal.h>
#include <sisci_types.h>
#include <sisci_api.h>


/* Local segment handle */
typedef struct {
    sci_desc_t          sd;
    unsigned            id;
    sci_local_segment_t seg;
    sci_map_t           map;
    const void*         ptr;
    size_t              len;
} segment_t;


/* Shared condition variable */
static pthread_mutex_t mtx_terminate;
static pthread_cond_t cv_terminate;
static bool terminate_cond = false;


/* Signal handler */
void terminate_program()
{
    pthread_mutex_lock(&mtx_terminate);
    terminate_cond = true;
    pthread_cond_signal(&cv_terminate);
    pthread_mutex_unlock(&mtx_terminate);
}


static int read_remote(int mdesc, int fdesc, size_t blk_sz, volatile void* ptr, size_t len)
{
    int rv;
    struct start_transfer request;

    request.file_desc = fdesc;
    request.block_size = blk_sz;
    request.num_blocks = len / blk_sz;
    request.file_pos = 0;
    request.remote_mem_ptr = (__u64) ptr;
    request.offset = 0;

    rv = ioctl(mdesc, SSD_DMA_START_TRANSFER, &request);
    if (rv < 0)
    {
        fprintf(stderr, "ioctl failed: %s\n", strerror(-rv));
    }

    return rv;
}


/* Local segment event */
static sci_callback_action_t segment_callback(void* data, sci_local_segment_t seg, sci_segment_cb_reason_t reason, unsigned node_id, unsigned seg_id, sci_error_t err)
{
    switch (reason)
    {
        case SCI_CB_CONNECT:
            fprintf(stderr, "Node %u connected to segment %u\n", node_id, seg_id);
            break;

        case SCI_CB_DISCONNECT:
            fprintf(stderr, "Node %u disconnected from segment %u\n", node_id, seg_id);
            break;

        default:
            fprintf(stderr, "Unexpected segment event for segment %u\n", seg_id);
            break;
    }

    return SCI_CALLBACK_CONTINUE;
}


static segment_t* create_segment(unsigned segment_id, unsigned adapter, size_t size)
{
    sci_error_t err;
    segment_t* ptr;

    if ((ptr = malloc(sizeof(segment_t))) == NULL)
    {
        fprintf(stderr, "Failed to allocate segment handle: %s\n", strerror(errno));
        return NULL;
    }

    ptr->id = segment_id;
    ptr->map = NULL;
    ptr->ptr = NULL;

    SCIOpen(&ptr->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open SISCI descriptor\n");
        goto leave;
    }

    SCICreateSegment(ptr->sd, &ptr->seg, segment_id, size, segment_callback, NULL, SCI_FLAG_USE_CALLBACK, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create local segment %u\n", segment_id);
        goto close;
    }

    SCIPrepareSegment(ptr->seg, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to prepare segment %u on adapter %u\n", segment_id, adapter);
        goto abort;
    }

    ptr->ptr = SCIMapLocalSegment(ptr->seg, &ptr->map, 0, size, NULL, SCI_FLAG_READONLY_MAP, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map local segment %u\n", segment_id);
        goto abort;
    }

    return ptr;

abort:
    SCIRemoveSegment(ptr->seg, 0, &err);

close:
    SCIClose(ptr->sd, 0, &err);

leave:
    free(ptr);

    return NULL;
}


static void remove_segment(segment_t* ptr)
{
    sci_error_t err;

    if (ptr != NULL)
    {
        do
        {
            SCIUnmapSegment(ptr->map, 0, &err);
        }
        while (err == SCI_ERR_BUSY);
        
        do
        {
            SCIRemoveSegment(ptr->seg, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        SCIClose(ptr->sd, 0, &err);

        free(ptr);
    }
}


static int convert_string(char* str, unsigned* result)
{
    char* ptr = NULL;

    *result = strtoul(str, &ptr, 0);
    if (ptr == NULL || *ptr != '\0')
    {
        return EINVAL;
    }

    return 0;
}


static int server(segment_t* segment, unsigned adapter)
{
    sci_error_t err;
    unsigned local_node = 0;

    SCIGetLocalNodeId(adapter, &local_node, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to get local node id\n");
    }

    SCISetSegmentAvailable(segment->seg, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to set segment available\n");
        return 1;
    }

    fprintf(stderr, "Connect to segment %u on node %u\n", segment->id, local_node);

    pthread_mutex_lock(&mtx_terminate);
    while (!terminate_cond)
    {
        pthread_cond_wait(&cv_terminate, &mtx_terminate);
    }
    pthread_mutex_unlock(&mtx_terminate);

    do
    {
        SCISetSegmentUnavailable(segment->seg, adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    return 0;
}


static int client(int module, int file, size_t block_size, unsigned remote_node, unsigned adapter, unsigned seg_id, size_t size)
{
    sci_error_t err;
    sci_desc_t sd;
    sci_remote_segment_t segment;
    size_t remote_size;
    volatile void* ptr;
    sci_map_t map;
    int ret = 0;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open descriptor\n");
        return 1;
    }

    SCIConnectSegment(sd, &segment, remote_node, seg_id, adapter, NULL, NULL, SCI_INFINITE_TIMEOUT, 0, &err);
    if (err != SCI_ERR_OK)
    {
        ret = 1;
        fprintf(stderr, "Failed to connect to remote segment %u on node %u\n", seg_id, remote_node);
        goto close_sisci;
    }

    remote_size = SCIGetRemoteSegmentSize(segment);
    if (remote_size < size)
    {
        size = remote_size;
    }

    ptr = SCIMapRemoteSegment(segment, &map, 0, size, NULL, 0, &err);
    if (ptr == NULL || err != SCI_ERR_OK)
    {
        ret = 2;
        fprintf(stderr, "Failed to map remote segment\n");
        goto disconnect;
    }

    if (read_remote(module, file, block_size, ptr, size) < 0)
    {
        fprintf(stderr, "Read remote failed\n");
    }
    
    do
    {
        SCIUnmapSegment(map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

disconnect:
    do
    {
        SCIDisconnectSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

close_sisci:
    SCIClose(sd, 0, &err);

    return ret;
}


int main(int argc, char** argv)
{
    sci_error_t err;

    struct option opts[] = {
        { .name = "adapter", .has_arg = 1, .flag = NULL, .val = 'a' },
        { .name = "rn", .has_arg = 1, .flag = NULL, .val = 'r' },
        { .name = "id", .has_arg = 1, .flag = NULL, .val = 'i' },
        { .name = "size", .has_arg = 1, .flag = NULL, .val = 's' },
        { .name = "file", .has_arg = 1, .flag = NULL, .val = 'f' },
        { .name = "block", .has_arg = 1, .flag = NULL, .val = 'b' },
        { .name = "help", .has_arg = 0, .flag = NULL, .val = 'h' }
    };

    segment_t* segment = NULL;
    int file_fd = -1;
    int module_fd = -1;

    unsigned adapter = 0;
    unsigned segment_id = 4;
    unsigned remote_node = 0;
    unsigned size = 0x1000;
    unsigned block_size = 512;
    const char* filename = NULL;

    int opt, idx;

    while ((opt = getopt_long(argc, argv, "-:a:r:i:s:f:b:h", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case ':':
                fprintf(stderr, "Missing value for option %s\n", argv[optind-1]);
                exit(':');

            case '?':
                fprintf(stderr, "Unknown option: %s\n", argv[optind-1]);
                exit('?');

            case 'h':
                // TODO show help
                exit('h');

            case 'a':
                if (convert_string(optarg, &adapter) != 0)
                {
                    fprintf(stderr, "Illegal adapter number: %s\n", optarg);
                    exit('a');
                }
                break;

            case 'r':
                if (convert_string(optarg, &remote_node) != 0)
                {
                    fprintf(stderr, "Illegal node id: %s\n", optarg);
                    exit('r');
                }
                break;

            case 'i':
                if (convert_string(optarg, &segment_id) != 0)
                {
                    fprintf(stderr, "Illegal segment id: %s\n", optarg);
                    exit('i');
                }
                break;

            case 's':
                if (convert_string(optarg, &size) != 0 || size == 0)
                {
                    fprintf(stderr, "Illegal size: %s\n", optarg);
                    exit('s');
                }
                break;

            case 'b':
                if (convert_string(optarg, &block_size) != 0)
                {
                    fprintf(stderr, "Illegal block size: %s\n", optarg);
                    exit('b');
                }

            case 'f':
                filename = optarg;
                break;
        }
    }

    SCIInitialize(0, &err);

    if (remote_node || filename)
    {
        //if ((file_fd = open(filename, O_DIRECT)) < 0)
        if ((file_fd = open(filename, O_RDONLY)) < 0)
        {
            fprintf(stderr, "Failed to open file: %s\n", strerror(errno));
            exit(2);
        }

        if ((module_fd = open(SSD_DMA_FILE_NAME, O_SYNC | O_RDONLY)) < 0)
        {
            close(file_fd);
            fprintf(stderr, "Failed to open fd to module: %s\n", strerror(errno));
            exit(3);
        }

        client(module_fd, file_fd, block_size, remote_node, adapter, segment_id, size);
        
        close(module_fd);
        close(file_fd);
        
    }
    else
    {
        signal(SIGTERM, (sig_t) terminate_program);
        signal(SIGINT, (sig_t) terminate_program);

        if ((segment = create_segment(segment_id, adapter, size)) == NULL)
        {
            fprintf(stderr, "Failed to create segment\n");
            exit(1);
        }
    
        server(segment, adapter);

        remove_segment(segment);
    }

    SCITerminate();
    return 0;
}
