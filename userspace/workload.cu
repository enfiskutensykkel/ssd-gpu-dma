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


__host__ __device__ static 
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


__global__ 
void poll_kernel(nvm_queue_t* queues, size_t n_cmds)
{
    nvm_queue_t* cq = &queues[0];
    struct completion* cpl;

    while (n_cmds > 0)
    {
        while ((cpl = cq_poll(cq)) == NULL);

        while ((cpl = cq_dequeue(cq)) != NULL)
        {
            // Retrieve SQ information from completion
            uint16_t sq_id = *CPL_SQID(cpl);
            uint16_t sq_hd = *CPL_SQHD(cpl);

            // Update submission queue
            nvm_queue_t* sq = &queues[sq_id];
            sq->head = sq_hd;

            --n_cmds;
        }

        // Update CQ head pointer to indicate that we're done processing the completion
        cq_update(cq);
    }
}


__global__ 
void work_kernel(nvm_queue_t* queues, buffer_t* data_ptr, size_t blk_size, size_t page_size, uint32_t ns_id, size_t n_cmds)
{
    //int num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int block = gridDim.x * blockIdx.y + blockIdx.x;
    int thread = blockDim.x * threadIdx.y + threadIdx.x;
    int id = blockDim.x * blockDim.y * block + thread;

    struct command* cmd;

    nvm_queue_t* sq = &queues[id + 1];

    //uint64_t virt_addr = ((uint64_t) data_ptr->virt_addr) + page_size * id;
    uint64_t bus_addr = bus_addr_mps(page_size * id, page_size, data_ptr->bus_addr);

    for (size_t i = 0; i < n_cmds; ++i)
    {
        while ((cmd = sq_enqueue(sq)) == NULL);

        prepare_read_cmd(cmd, blk_size, page_size, ns_id, 0, 1, NULL, NULL, &bus_addr);;
    }

    sq_submit(sq);
}


__host__ static
int launch_kernel(const nvm_ctrl_t* ctrl, nvm_queue_t* queues, buffer_t* data, int n_threads, uint32_t ns_id)
{
    cudaError_t err;

    cudaStream_t stream;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        return ENOSPC;
    }

    nvm_queue_t* dev_queues;
    err = cudaMalloc(&dev_queues, sizeof(nvm_queue_t) * (n_threads + 1));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        return ENOMEM;
    }

    buffer_t* dev_data;
    err = cudaMalloc(&dev_data, sizeof(buffer_t) + sizeof(uint64_t) * data->n_addrs);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(dev_queues);
        return ENOMEM;
    }

    err = cudaMemcpy(dev_data, data, sizeof(buffer_t) + sizeof(uint64_t) * data->n_addrs, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy descriptor to device: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(dev_queues);
        cudaFree(dev_data);
        return EIO;
    }

    err = cudaMemcpy(dev_queues, queues, sizeof(nvm_queue_t) * (n_threads + 1), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy queue descriptors to device: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(dev_queues);
        cudaFree(dev_data);
        return EIO;
    }

    // TODO: Use MSI-X interrupts to do something meaningful

    // Start kernels
    poll_kernel<<<1, 1>>>(dev_queues, /*n_threads*/ 1); 

    // TODO: record event before and after

    work_kernel<<<1, n_threads, 0, stream>>>(dev_queues, dev_data, 512, ctrl->page_size, ns_id, 2);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize CUDA stream: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize device: %s\n", cudaGetErrorString(err));
    }

    uint32_t result = 0xcafebabe;
    cudaMemcpy(&result, data->virt_addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%x\n", result);

    cudaStreamDestroy(stream);
    cudaFree(dev_queues);
    cudaFree(dev_data);

    return 0;
}


__host__ static 
int create_submission_queues(nvm_ctrl_t* ctrl, buffer_t* queue_mem, nvm_queue_t* queue_descs, int n_threads, void* reg_ptr)
{
    uint64_t virt_addr;
    uint64_t bus_addr;
    int err;
    int i;

    const nvm_queue_t* cq = &queue_descs[0];

    // Create submission queues
    for (i = 1; i <= n_threads; ++i)
    {
        nvm_queue_t* sq = &queue_descs[i];

        bus_addr = bus_addr_mps(ctrl->page_size * i, queue_mem->page_size, queue_mem->bus_addr);
        virt_addr = ((uint64_t) queue_mem->virt_addr) + ctrl->page_size * i;

        err = nvm_create_sq(ctrl, cq, sq, i, (void*) virt_addr, bus_addr, reg_ptr);
        if (err != 0)
        {
            fprintf(stderr, "Failed to create submission queue %d of %d: %s\n", i + 1, n_threads, strerror(err));
            return i - 1;
        }
    }

    return i - 1;
}


extern "C" __host__
int cuda_workload(int ioctl_fd, nvm_ctrl_t* ctrl, int dev, uint32_t ns_id, void* io_mem, size_t io_size)
{
    int n_threads = 2;
    int status;
    buffer_t* queue_memory = NULL;
    buffer_t* data_buffer = NULL;
    nvm_queue_t* queue_descriptors = NULL;

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
    void* reg_ptr;
    err = cudaHostGetDevicePointer(&reg_ptr, io_mem, 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to map IO memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto unregister;
    }

    // Allocate queue memory
    queue_memory = get_buffer(ioctl_fd, dev, ctrl->page_size * (n_threads + 1));
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
        goto free_queue_memory;
    }

    // Allocate submission queue descriptors
    err = cudaHostAlloc(&queue_descriptors, sizeof(nvm_queue_t) * (n_threads + 1), cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate handle memory: %s\n", cudaGetErrorString(err));
        status = ENOMEM;
        goto free_queue_memory;
    }

    // Allocate data buffer
    data_buffer = get_buffer(ioctl_fd, dev, 0x1000 * n_threads);
    if (data_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        status = ENOMEM;
        goto free_descriptors;
    }

    status = nvm_create_cq(ctrl, queue_descriptors, 1, queue_memory->virt_addr, queue_memory->bus_addr[0], reg_ptr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create IO completion queue: %s\n", strerror(status));
        goto free_descriptors;
    }

    // Create IO queue pairs
    if (create_submission_queues(ctrl, queue_memory, queue_descriptors, n_threads, reg_ptr) != n_threads)
    {
        fprintf(stderr, "Failed to create IO submission queues\n");
        status = ENOMEM;
        goto free_data_buffer;
    }

    // Do some work
    status = launch_kernel(ctrl, queue_descriptors, data_buffer, n_threads, ns_id);

    // Clean up and exit

    // TODO delete_sqs:
    // TODO delete_cq

free_data_buffer:
    put_buffer(ioctl_fd, data_buffer);

free_descriptors:
    cudaFreeHost(queue_descriptors);

free_queue_memory:
    put_buffer(ioctl_fd, queue_memory);

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
