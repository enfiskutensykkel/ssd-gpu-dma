#include <cuda.h>
#include "memory.h"
#include "nvm/types.h"
#include "nvm/queue.h"
#include "nvm/command.h"
#include "nvm/util.h"
#include "nvm/ctrl.h"
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>


__device__ 
int prepare_read_cmd(struct command* cmd, uint32_t blk_size, uint32_t page_size, uint32_t ns_id, uint64_t start_lba, uint16_t n_blks, void* list_virt, const uint64_t* list, const uint64_t* data)
{
    size_t blks_per_page = page_size / blk_size;
    size_t n_prps = n_blks / blks_per_page;

    if (n_blks < 1)
    {
        return ERANGE;
    }
    else if (n_prps > 2 && list == NULL)
    {
        return EINVAL;
    }

    uint64_t prp1 = data[0];
    uint64_t prp2 = 0;
    if (1 < n_prps && n_prps <= 2)
    {
        prp2 = data[1];
    }
    else if (n_prps > 2)
    {
        prp2 = build_prp_list(page_size, list_virt, n_prps - 1, list, data + 1);
    }

    cmd_header(cmd, NVM_READ, ns_id);
    cmd_data_ptr(cmd, prp1, prp2);
    
    cmd->dword[10] = (uint32_t) start_lba;
    cmd->dword[11] = (uint32_t) (start_lba >> 32);

    cmd->dword[12] = (0x00 << 31) | (0x00 << 30) | (0x00 << 26) | n_blks;

    cmd->dword[13] = 0;
    cmd->dword[14] = 0;
    cmd->dword[15] = 0;

    return 0;
}


__device__ inline
unsigned thread_pos()
{
    //int n_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    unsigned block_id = gridDim.x * blockIdx.y + blockIdx.x;
    unsigned thread_id = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned thread_pos = blockDim.x * blockDim.y * block_id + thread_id;

    return thread_pos;
}


__global__ 
void produce_kernel(buffer_t* buffer, nvm_queue_t* queues, size_t blk_size, size_t page_size, uint32_t ns_id, unsigned n_cmds)
{
    unsigned thread_id = thread_pos();

    nvm_queue_t* sq = &queues[2 * thread_id + 1]; 
    unsigned enqueued_cmds = 0;

    uint32_t transfer_pages = 1;
    uint64_t* bus_addr = &buffer->bus_addr[thread_id * transfer_pages];
    
    // Produce commands
    while (enqueued_cmds < n_cmds)
    {
        bool prepared = false;
        struct command* cmd;
        while ((cmd = sq_enqueue(sq)) != NULL)
        {
            if (prepare_read_cmd(cmd, blk_size, page_size, ns_id, 0, 2, NULL, NULL, bus_addr) == 0)
            {
                ++enqueued_cmds;
                prepared = true;
            }
        }

        // Only write to doorbell if there are any commands ready
        if (prepared)
        {
            sq_submit(sq);
        }
    }
}


__global__
void consume_kernel(nvm_queue_t* queues, unsigned n_cpls)
{
    unsigned thread_id = thread_pos();

    nvm_queue_t* cq = &queues[2 * thread_id];
    nvm_queue_t* sq = &queues[2 * thread_id + 1]; 
    unsigned dequeued_cpls = 0;

    // Consume completions
    while (dequeued_cpls < n_cpls)
    {
        struct completion* cpl;
        while ((cpl = cq_dequeue(cq)) != NULL)
        {
            uint16_t sq_hd = *CPL_SQHD(cpl);

            sq->head = sq_hd;
            ++dequeued_cpls;

            // Only update doorbell if head changes
            if (dequeued_cpls % sq->max_entries == sq_hd)
            {
                cq_update(cq);
            }
        }
    }
}



__host__ 
int launch_kernel(const nvm_ctrl_t* ctrl, nvm_queue_t* queues, size_t n_tasks, buffer_t* buffer, uint32_t ns_id, unsigned cmds_per_task)
{
    cudaError_t err;
    nvm_queue_t* dev_queues = NULL;
    buffer_t* dev_buffer = NULL;
    cudaStream_t producer_stream;
    cudaStream_t consumer_stream;
    int status = 0;

    err = cudaMalloc(&dev_queues, sizeof(nvm_queue_t) * n_tasks * 2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate descriptor memory on device: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto exit;
    }

    err = cudaMemcpy(dev_queues, queues, sizeof(nvm_queue_t) * n_tasks * 2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy queue descriptors to device: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    err = cudaMalloc(&dev_buffer, sizeof(buffer_t) + sizeof(uint64_t) * buffer->n_addrs);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate descriptor memory on device: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto exit;
    }

    err = cudaMemcpy(dev_buffer, buffer, sizeof(buffer_t) + sizeof(uint64_t) * buffer->n_addrs, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy buffer descriptor to device: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    err = cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stream: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    err = cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        cudaStreamDestroy(producer_stream);
        fprintf(stderr, "Failed to create stream: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    // Run GPU threads
    consume_kernel<<<1, n_tasks, 0, consumer_stream>>>(dev_queues, cmds_per_task);

    fprintf(stderr, "Running %zu threads each sending %u commands...\n", n_tasks, cmds_per_task);
    produce_kernel<<<1, n_tasks, 0, producer_stream>>>(dev_buffer, dev_queues, 512, ctrl->page_size, ns_id, cmds_per_task);

    // Wait for streams to finish
    fprintf(stderr, "Waiting for kernel completion...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize device: %s\n", cudaGetErrorString(err));
    }

    cudaStreamDestroy(producer_stream);
    cudaStreamDestroy(consumer_stream);

    fprintf(stderr, "Done\n");

exit:
    cudaFree(dev_queues);
    cudaFree(dev_buffer);
    return status;
}


//__host__ static
//void remove_tasks(nvm_queue_t* queues, cudaStream_t* streams, size_t n_tasks)
//{
//    cudaError_t err;
//
//    for (size_t task = 0; task < n_tasks; ++task)
//    {
//        // TODO: Submit delete submission queue
//    }
//}


__host__ static 
size_t create_tasks(nvm_ctrl_t* ctrl, buffer_t* queue_memory, nvm_queue_t* queues, size_t n_tasks, void* reg_ptr)
{
    size_t task;

    for (task = 0; task < n_tasks; ++task)
    {
        nvm_queue_t* cq = &queues[2 * task];
        
        uint64_t ioaddr = queue_memory->bus_addr[2 * task]; // FIXME: use same calculation as in pin_memory()
        uint64_t vaddr = ((uint64_t) queue_memory->virt_addr) + ctrl->page_size * (2 * task);

        int status = nvm_create_cq(ctrl, cq, task + 1, (void*) vaddr, ioaddr, reg_ptr);
        if (status != 0)
        {
            fprintf(stderr, "Failed to create completion queue %zu of %zu: %s\n", task + 1, n_tasks, strerror(status));
            return task - 1;
        }

        nvm_queue_t* sq = &queues[2 * task + 1];

        ioaddr = queue_memory->bus_addr[2 * task + 1];
        vaddr = ((uint64_t) queue_memory->virt_addr) + ctrl->page_size * (2 * task + 1);

        status = nvm_create_sq(ctrl, cq, sq, task + 1, (void*) vaddr, ioaddr, reg_ptr);
        if (status != 0)
        {
            fprintf(stderr, "Failed to create submission queue %zu of %zu: %s\n", task + 1, n_tasks, strerror(status));
            return task - 1;
        }
    }

    return task;
}


extern "C" __host__
int cuda_workload(int ioctl_fd, nvm_ctrl_t* ctrl, int dev, uint32_t ns_id, void* io_mem, size_t io_size, size_t n_tasks, unsigned n_cmds)
{
    size_t created;
    int status;
    buffer_t* queue_memory = NULL;
    buffer_t* cq_memory = NULL;
    buffer_t* data_buffer = NULL;
    nvm_queue_t* queues = NULL;
    void* reg_ptr = NULL;

    // Set CUDA device
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
        status = EINVAL;
        goto exit;
    }

    // Map IO memory into CUDA address space
    err = cudaHostRegister(io_mem, io_size, cudaHostRegisterIoMemory);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to map IO memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    // Get device pointer to mapped IO memory
    err = cudaHostGetDevicePointer(&reg_ptr, io_mem, 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to map IO memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto unregister;
    }

    // Allocate queue memory
    queue_memory = get_buffer(ioctl_fd, dev, ctrl->page_size * n_tasks * 2, ctrl->page_size);
    if (queue_memory == NULL)
    {
        status = ENOMEM;
        goto unregister;
    }

    // Zero out queue memory
    err = cudaMemset(queue_memory->virt_addr, 0, queue_memory->range_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto free_memory;
    }

    // Allocate queue descriptor array
    err = cudaHostAlloc(&queues, sizeof(nvm_queue_t) * n_tasks * 2, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate handle memory: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto free_memory;
    }

    // Allocate data buffer
    data_buffer = get_buffer(ioctl_fd, dev, ctrl->page_size * n_tasks, ctrl->page_size); 
    if (data_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        status = ENOMEM;
        goto free_memory;
    }

    //cudaMemAdvice(data_buffer->virt_addr, dat_buffer->range_size, cuda

    // Create tasks
    created = create_tasks(ctrl, queue_memory, queues, n_tasks, reg_ptr);
    if (created != n_tasks)
    {
        fprintf(stderr, "Failed to create tasks\n");
        status = EIO;
        //goto free_tasks;
        goto free_memory;
    }

    // Do some work
    status = launch_kernel(ctrl, queues, n_tasks, data_buffer, ns_id, n_cmds);

    // TODO: send abort command

//free_tasks:
    //remove_tasks(queues, created);

free_memory:
    put_buffer(ioctl_fd, data_buffer);
    cudaFreeHost(queues);
    put_buffer(ioctl_fd, queue_memory);
    put_buffer(ioctl_fd, cq_memory);

unregister:
    cudaHostUnregister(io_mem);

exit:
    //cudaDeviceReset();
    return status;
}

//__host__ __device__
//static int prepare_write_cmd(nvm_queue_t sq, uint32_t ns_id, uint32_t blk_size, memory_t* buffer, uint64_t start_lba, uint16_t n_blks)
//{
//    struct command* cmd = sq_enqueue(sq);
//    if (cmd == NULL)
//    {
//        return EAGAIN;
//    }
//
//    cmd_header(cmd, NVM_WRITE, ns_id);
//    cmd_data_ptr(cmd, NULL, buffer, 1);
//    cmd->dword[10] = (uint32_t) start_lba;
//    cmd->dword[11] = (uint32_t) (start_lba >> 1);
//
//    cmd->dword[12] = n_blks;
//    cmd->dword[13] = 0;
//    cmd->dword[14] = 0;
//    cmd->dword[15] = 0;
//    return 0;
//}
