#include <cuda.h>
#include "memory.h"
#include "nvm/types.h"
#include "nvm/queue.h"
#include "nvm/command.h"
#include "nvm/util.h"
#include "nvm/ctrl.h"
#include <pthread.h>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>


struct consumer_data
{
    nvm_queue_t*        queues;
    size_t*             n_reqs;
    bool                all_complete;
};



/* Dequeue completions from completion queue,
 * update submission queues and number of requests.
 */
__host__ static
void* consumer(struct consumer_data* data)
{
    struct completion* cpl;
    nvm_queue_t* cq = &data->queues[0];

    // Run until all streams are completed
    while (!data->all_complete)
    {
        // Consume completions
        while ((cpl = cq_dequeue(cq)) != NULL)
        {
            uint16_t sq_id = *CPL_SQID(cpl);
            uint16_t sq_hd = *CPL_SQHD(cpl);

            nvm_queue_t* sq = &data->queues[sq_id];
            sq->head = sq_hd;

            data->n_reqs[0]++;
            data->n_reqs[sq_id]++;

            if (data->n_reqs[sq_id] % sq->max_entries == sq_hd)
            {
                cq_update(cq);
            }
        }
    }

    return NULL;
}


__device__ static 
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


__global__ static
void work_kernel(size_t task_id, buffer_t* buffer, nvm_queue_t* queues, size_t blk_size, size_t page_size, uint32_t ns_id)
{
    nvm_queue_t* sq = &queues[task_id]; 

    bool prepared = false;

    // Enqueue commands
    struct command* cmd;
    while ((cmd = sq_enqueue(sq)) != NULL)
    {
        if (prepare_read_cmd(cmd, blk_size, page_size, ns_id, 0, 1, NULL, NULL, buffer->bus_addr) == 0)
        {
            prepared = true;
        }
    }

    if (prepared)
    {
        sq_submit(sq);
    }
}


__host__ static
int launch_kernel(const nvm_ctrl_t* ctrl, nvm_queue_t* queues, cudaStream_t* streams, size_t n_tasks, buffer_t* buffer, uint32_t ns_id, size_t cmds_per_task)
{
    cudaError_t err;
    nvm_queue_t* dev_queues = NULL;
    size_t* n_reqs = NULL;
    buffer_t* dev_buffer = NULL;
    int status = 0;
    pthread_t thread;
    struct consumer_data consumer_data;

    err = cudaMalloc(&dev_queues, sizeof(nvm_queue_t) * n_tasks);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate descriptor memory on device: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto exit;
    }

    err = cudaMemcpy(dev_queues, &queues[1], sizeof(nvm_queue_t) * n_tasks, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy queue descriptors to device: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto exit;
    }

    err = cudaHostAlloc(&n_reqs, sizeof(size_t) * (n_tasks + 1), cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate request counter: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto exit;
    }
    memset(n_reqs, 0, sizeof(size_t) * (n_tasks + 1));

    err = cudaMalloc(&dev_buffer, sizeof(buffer_t) + sizeof(uint64_t) * buffer->n_addrs);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate descriptor memory on device: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto exit;
    }

    // Start dequeuer thread
    consumer_data.queues = queues;
    consumer_data.n_reqs = n_reqs;
    consumer_data.all_complete = false;

    status = pthread_create(&thread, NULL, (void* (*)(void*)) consumer, &consumer_data);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create thread: %s\n", strerror(status));
        goto exit;
    }

    // Queue kernels on their respective streams
    fprintf(stderr, "Running %zu kernels each sending %zu commands...\n", n_tasks, cmds_per_task);
    while (!consumer_data.all_complete)
    {
        bool all_complete = true;

        for (size_t task = 0; task < n_tasks; ++task)
        {
            nvm_queue_t* host_queue = &queues[task + 1];
            unsigned char* dev_queue = (unsigned char*) &dev_queues[task];
            cudaStream_t stream = streams[task];
            size_t* requests = &n_reqs[task + 1];

            // Check if stream is complete
            if (*requests >= cmds_per_task)
            {
                continue;
            }

            all_complete = false;

            // Update the queue descriptor for the GPU thread so it is able to enqueue more commands
            uint32_t head = host_queue->head;
            err = cudaMemcpyAsync(dev_queue + offsetof(nvm_queue_t, head), &head, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy head pointer (%zu): %s\n", task, cudaGetErrorString(err));
                all_complete = true;
                break;
            }

            // Start a kernel on the stream
            work_kernel<<<1, 1, 0, stream>>>(task, dev_buffer, dev_queues, 512, ctrl->page_size, ns_id);
        }

        consumer_data.all_complete = all_complete;
    }

    // Wait for streams to finish
    fprintf(stderr, "Waiting for kernels...\n");
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize device: %s\n", cudaGetErrorString(err));
    }

    // Wait for completion consumer to finish
    status = pthread_join(thread, NULL);
    if (status != 0)
    {
        fprintf(stderr, "Failed to join thread: %s\n", strerror(status));
    }

    fprintf(stderr, "%zu kernels sent %zu commands\n", n_tasks, n_reqs[0]);

exit:
    //cudaFree(queue_heads);
    cudaFreeHost(n_reqs);
    cudaFree(dev_queues);
    cudaFree(dev_buffer);
    return status;
}


__host__ static
void remove_tasks(nvm_queue_t* queues, cudaStream_t* streams, size_t n_tasks)
{
    cudaError_t err;

    for (size_t task = 0; task < n_tasks; ++task)
    {
        err = cudaStreamDestroy(streams[task]);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to destroy stream: %s\n", cudaGetErrorString(err));
        }

        // TODO: Submit delete submission queue
    }
}


__host__ static 
size_t create_tasks(nvm_ctrl_t* ctrl, buffer_t* queue_memory, nvm_queue_t* queues, cudaStream_t* streams, size_t n_tasks, void* reg_ptr)
{
    cudaError_t err;
    size_t task;

    const nvm_queue_t* cq = &queues[0];
    for (task = 0; task < n_tasks; ++task)
    {
        cudaStream_t* stream = &streams[task];
        nvm_queue_t* sq = &queues[task + 1];

        err = cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to create stream: %s\n", cudaGetErrorString(err));
            return task - 1;
        }

        uint64_t bus_addr = queue_memory->bus_addr[task]; // FIXME: use same calculation as in pin_memory()
        uint64_t virt_addr = ((uint64_t) queue_memory->virt_addr) + ctrl->page_size * task;

        int status = nvm_create_sq(ctrl, cq, sq, task + 1, (void*) virt_addr, bus_addr, reg_ptr);
        if (status != 0)
        {
            fprintf(stderr, "Failed to create submission queue %zu of %zu: %s\n", task + 1, n_tasks, strerror(status));
            cudaStreamDestroy(*stream);
            return task - 1;
        }
    }

    return task;
}


extern "C" __host__
int cuda_workload(int ioctl_fd, nvm_ctrl_t* ctrl, int dev, uint32_t ns_id, void* io_mem, size_t io_size, size_t n_tasks, size_t n_cmds)
{
    size_t created;
    int status;
    buffer_t* queue_memory = NULL;
    buffer_t* cq_memory = NULL;
    buffer_t* data_buffer = NULL;
    nvm_queue_t* queues = NULL;
    void* reg_ptr = NULL;
    cudaStream_t* streams = NULL;

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
    queue_memory = get_buffer(ioctl_fd, dev, ctrl->page_size * n_tasks, ctrl->page_size);
    if (queue_memory == NULL)
    {
        status = ENOMEM;
        goto unregister;
    }

    cq_memory = get_buffer(ioctl_fd, -1, ctrl->page_size, ctrl->page_size);
    if (cq_memory == NULL)
    {
        status = ENOMEM;
        goto free_memory;
    }
    memset(cq_memory->virt_addr, 0, cq_memory->range_size);

    // Zero out queue memory
    err = cudaMemset(queue_memory->virt_addr, 0, queue_memory->range_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto free_memory;
    }

    // Allocate task memory
    err = cudaHostAlloc(&streams, sizeof(cudaStream_t) * n_tasks, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate stream descriptors: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto free_memory;
    }

    // Allocate submission queue descriptors
    err = cudaHostAlloc(&queues, sizeof(nvm_queue_t) * (n_tasks + 1), cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate handle memory: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto free_memory;
    }

    // Allocate data buffer
    data_buffer = get_buffer(ioctl_fd, dev, ctrl->page_size, ctrl->page_size); 
    if (data_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        status = ENOMEM;
        goto free_memory;
    }

    //cudaMemAdvice(data_buffer->virt_addr, dat_buffer->range_size, cuda

    // Create completion queue
    status = nvm_create_cq(ctrl, queues, 1, cq_memory->virt_addr, cq_memory->bus_addr[0], ctrl->reg_ptr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create IO completion queue: %s\n", strerror(status));
        goto free_memory;
    }

    // Create tasks
    created = create_tasks(ctrl, queue_memory, queues, streams, n_tasks, reg_ptr);
    if (created != n_tasks)
    {
        fprintf(stderr, "Failed to create tasks\n");
        status = EIO;
        goto free_tasks;
    }

    // Do some work
    status = launch_kernel(ctrl, queues, streams, n_tasks, data_buffer, ns_id, n_cmds);

    // TODO: send abort command

free_tasks:
    remove_tasks(queues, streams, created);

free_memory:
    put_buffer(ioctl_fd, data_buffer);
    cudaFreeHost(queues);
    put_buffer(ioctl_fd, queue_memory);
    put_buffer(ioctl_fd, cq_memory);

unregister:
    cudaHostUnregister(io_mem);

exit:
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
