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

//
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

__host__ __device__ static 
int prepare_read_cmd(nvm_queue_t* sq, uint32_t blk_size, uint32_t page_size, uint32_t ns_id, uint64_t start_lba, uint16_t n_blks, void* list_virt, const uint64_t* list, const uint64_t* data)
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

    struct command* cmd = sq_enqueue(sq);
    if (cmd == NULL)
    {
        return EAGAIN;
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
void work_kernel(nvm_queue_t* queues, buffer_t* data_ptr, size_t blk_size, size_t page_size, uint32_t ns_id)
{
    int num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int block = gridDim.x * blockIdx.y + blockIdx.x;
    int thread = blockDim.x * threadIdx.y + threadIdx.x;
    int id = blockDim.x * blockDim.y * block + thread;

    nvm_queue_t* sq = &queues[id];

    uint64_t virt_addr = ((uint64_t) data_ptr->virt_addr) + page_size * id;
    uint64_t bus_addr = bus_addr_mps(page_size * id, page_size, data_ptr->bus_addr);

    if (prepare_read_cmd((nvm_queue_t*) sq, blk_size, page_size, ns_id, 0, 1, NULL, NULL, &bus_addr) != 0)
    {
        return;
    }

    sq_submit((nvm_queue_t*) sq);

    // FIXME: get hold of updated SQ head somehow
}


__host__ static inline 
cudaError_t update_sq(nvm_queue_t* dev_ptr, uint16_t head)
{
    // FIXME: this doesn't work, probably because of some alignment stuff
    //return cudaMemcpyAsync(((unsigned char*) dev_ptr) + offsetof(nvm_queue_t, head), &head, sizeof(uint16_t), cudaMemcpyHostToDevice);

    return cudaSuccess;
}


__host__ static
int launch_kernel(const nvm_ctrl_t* ctrl, nvm_queue_t* cq, nvm_queue_t* queues, buffer_t* data, int n_threads, uint32_t ns_id)
{
    cudaError_t err;

    nvm_queue_t* dev_queues;
    err = cudaMalloc(&dev_queues, sizeof(nvm_queue_t) * n_threads);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return ENOMEM;
    }

    buffer_t* dev_data;
    err = cudaMalloc(&dev_data, sizeof(buffer_t) + sizeof(uint64_t) * data->n_addrs);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        cudaFree(dev_queues);
        return ENOMEM;
    }

    err = cudaMemcpy(dev_data, data, sizeof(buffer_t) + sizeof(uint64_t) * data->n_addrs, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy descriptor to device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_queues);
        cudaFree(dev_data);
        return EIO;
    }

    err = cudaMemcpy(dev_queues, queues, sizeof(nvm_queue_t) * n_threads, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy queue descriptors to device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_queues);
        cudaFree(dev_data);
        return EIO;
    }

    // Start kernel
    work_kernel<<<1, n_threads>>>(dev_queues, dev_data, 512, ctrl->page_size, ns_id);

    // Start waiting for completions
    int thread_count = 0;
    struct completion* cpl;
    
    // TODO: Host completion queue on GPU memory instead (one queue is fine) and use MSI from host

    while (thread_count < n_threads)
    {
        while ((cpl = cq_dequeue(cq)) == NULL);

        ++thread_count;

        // Retrieve SQ info
        uint16_t sq_id = *CPL_SQID(cpl);
        uint16_t sq_hd = *CPL_SQHD(cpl);

        // Update CQ head pointer
        cq_update(cq);

        // Update SQ head pointers on the fly
        err = update_sq(dev_queues + sq_id - 1, sq_hd);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to update device's SQ: %s\n", cudaGetErrorString(err));
        }
    }

    uint32_t result = 0xcafebabe;
    cudaMemcpy(&result, data->virt_addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%x\n", result);

    cudaFree(dev_queues);
    cudaFree(dev_data);
    return 0;
}


__host__ static 
int create_submission_queues(nvm_ctrl_t* ctrl, const nvm_queue_t* cq, buffer_t* queue_mem, nvm_queue_t* queue_descs, int n_threads, void* reg_ptr)
{
    uint64_t virt_addr;
    uint64_t bus_addr;
    int err;
    int i;

    // Create submission queues
    for (i = 0; i < n_threads; ++i)
    {
        nvm_queue_t* sq = &queue_descs[i];

        bus_addr = bus_addr_mps(ctrl->page_size * i, queue_mem->page_size, queue_mem->bus_addr);
        virt_addr = ((uint64_t) queue_mem->virt_addr) + ctrl->page_size * i;

        err = nvm_create_sq(ctrl, cq, sq, i + 1, (void*) virt_addr, bus_addr, reg_ptr);
        if (err != 0)
        {
            fprintf(stderr, "Failed to create submission queue %d of %d: %s\n", i + 1, n_threads, strerror(err));
            return i;
        }
    }

    return i;
}


extern "C" __host__
int cuda_workload(int ioctl_fd, nvm_ctrl_t* ctrl, int dev, uint32_t ns_id, void* io_mem, size_t io_size)
{
    int n_threads = 2;
    int status;
    buffer_t* queue_memory = NULL;
    buffer_t* data_buffer = NULL;
    nvm_queue_t* queue_descriptors = NULL;
    page_t cq_mem;
    nvm_queue_t cq;

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

    // Allocate CQ memory
    status = get_page(ioctl_fd, -1, &cq_mem);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate queue memory: %s\n", strerror(status));
        goto unregister;
    }
    memset(cq_mem.virt_addr, 0, cq_mem.page_size);

    // Allocate SQ memory
    queue_memory = get_buffer(ioctl_fd, dev, ctrl->page_size * 2 * n_threads);
    if (queue_memory == NULL)
    {
        status = ENOMEM;
        goto free_page;
    }
    
    // Zero out queue memory
    err = cudaMemset(queue_memory->virt_addr, 0, queue_memory->range_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set device memory: %s\n", cudaGetErrorString(err));
        status = EIO;
        goto free_queue_memory;
    }

    status = nvm_create_cq(ctrl, &cq, 1, cq_mem.virt_addr, cq_mem.bus_addr, ctrl->reg_ptr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create IO completion queue: %s\n", strerror(status));
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

    // Allocate PRP list ranges
    //err = cudaHostAlloc(


    // Allocate data buffer
    data_buffer = get_buffer(ioctl_fd, dev, 0x1000 * n_threads);
    if (data_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        status = ENOMEM;
        goto free_descriptors;
    }

    // Create IO queue pairs
    if (create_submission_queues(ctrl, &cq, queue_memory, queue_descriptors, n_threads, reg_ptr) != n_threads)
    {
        status = ENOMEM;
        goto free_data_buffer;
    }

    // Do some work
    status = launch_kernel(ctrl, &cq, queue_descriptors, data_buffer, n_threads, ns_id);

    // Clean up and exit

    // TODO delete_sqs:
    // TODO delete_cq

free_data_buffer:
    put_buffer(ioctl_fd, data_buffer);

free_descriptors:
    cudaFreeHost(queue_descriptors);

free_queue_memory:
    put_buffer(ioctl_fd, queue_memory);

free_page:
    put_page(ioctl_fd, &cq_mem);

unregister:
    cudaHostUnregister(io_mem);

exit:
    return status;
}

