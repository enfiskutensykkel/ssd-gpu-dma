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


//
//
//
//__global__ void do_work(memory_t* buffer, nvm_queue_t sq, uint32_t* tailst) //, nvm_queue_t cq)
//{
//    *tailst = 0;
//    if (prepare_read_cmd(sq, 1, 512, buffer, 0, 1) == 0)
//    {
//        *tailst = sq->tail;
//        sq_submit(sq); // this works
//    }
//
//    //while (cq_poll(cq) == NULL);
//
//    //*result = *((uint32_t*) buffer->virt_addr);
//}
//

__global__ static
void work_kernel(nvm_queue_t* queues, buffer_t* data_ptr, size_t blk_size, size_t page_size, uint32_t ns_id)
{
    int num = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    int block = gridDim.x * blockIdx.y + blockIdx.x;
    int thread = blockDim.x * threadIdx.y + threadIdx.x;
    int id = blockDim.x * blockDim.y * block + thread;

    nvm_queue_t* sq = &queues[id * 2];
    nvm_queue_t* cq = &queues[id * 2 + 1];

    uint64_t virt_addr = ((uint64_t) data_ptr->virt_addr) + page_size * id;
    uint64_t bus_addr = bus_addr_mps(page_size * id, page_size, data_ptr->bus_addr);

    if (prepare_read_cmd(&queues[id * 2], blk_size, page_size, ns_id, 0, 1, NULL, NULL, &bus_addr) != 0)
    {
        __sync_threads();
        return;
    }

    sq_submit(&queues[id * 2]);
}


__host__ static
int launch_kernel(const nvm_ctrl_t* ctrl, nvm_queue_t* queues, buffer_t* data, int n_threads, uint32_t ns_id)
{
    cudaError_t err;

    nvm_queue_t* dev_queues;
    err = cudaMalloc(&dev_queues, sizeof(nvm_queue_t) * 2 * n_threads);
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

    err = cudaMemcpy(dev_queues, queues, sizeof(nvm_queue_t) * 2 * n_threads, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy queue descriptors to device: %s\n", cudaGetErrorString(err));
        cudaFree(dev_queues);
        cudaFree(dev_data);
        return EIO;
    }


    work_kernel<<<1, 1>>>(dev_queues, dev_data, 512, ctrl->page_size, ns_id);


    struct completion cpl;
    cudaMemcpy(&cpl, queues[1].virt_addr, sizeof(struct completion), cudaMemcpyDeviceToHost);

    fprintf(stderr, "cid=%u sct=%x sc=%x\n", *CPL_CID(&cpl), SCT(&cpl), SC(&cpl));

    uint32_t result = 0xcafebabe;
    cudaMemcpy(&result, data->virt_addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    fprintf(stderr, "%x\n", result);

    cudaFree(dev_queues);
    cudaFree(dev_data);
    return 0;
}


__host__ static 
int create_queues(nvm_ctrl_t* ctrl, buffer_t* queue_mem, nvm_queue_t* queue_ds, int n_threads, void* reg_ptr)
{
    uint64_t virt_addr;
    uint64_t bus_addr;
    int err;

    for (int i = 0; i < n_threads; ++i)
    {
        // Create completion queue
        nvm_queue_t* cq = &queue_ds[i * 2 + 1];
        bus_addr = bus_addr_mps(ctrl->page_size * (i * 2 + 1), queue_mem->page_size, queue_mem->bus_addr);
        virt_addr = ((uint64_t) queue_mem->virt_addr) + ctrl->page_size * (i * 2 + 1);

        err = nvm_create_cq(cq, i + 1, ctrl, (void*) virt_addr, bus_addr, reg_ptr);
        if (err != 0)
        {
            fprintf(stderr, "Failed to create completion queue: %s\n", strerror(err));
            return err;
        }

        // Create submission queue
        nvm_queue_t* sq = &queue_ds[i * 2];
        bus_addr = bus_addr_mps(ctrl->page_size * (i * 2 + 1), queue_mem->page_size, queue_mem->bus_addr);
        virt_addr = ((uint64_t) queue_mem->virt_addr) + ctrl->page_size * (i * 2 + 1);

        err = nvm_create_sq(sq, i + 1, ctrl, (void*) virt_addr, bus_addr, reg_ptr);
        if (err != 0)
        {
            fprintf(stderr, "Failed to create submission queue: %s\n", strerror(err));
            return err;
        }
    }

    return 0;
}


extern "C" __host__
int cuda_workload(int ioctl_fd, nvm_ctrl_t* ctrl, int dev, uint32_t ns_id, void* io_mem, size_t io_size)
{
    int n_threads = 16;
    int status;
    buffer_t* queue_memory = NULL;
    buffer_t* data_buffer = NULL;
    nvm_queue_t* queue_descriptors = NULL;

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
        goto exit;
    }

    // Allocate queue memory
    queue_memory = get_buffer(ioctl_fd, dev, ctrl->page_size * 2 * n_threads);
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
        goto queue_memory;
    }

    // Allocate queue descriptors
    err = cudaHostAlloc(&queue_descriptors, sizeof(nvm_queue_t) * 2 * n_threads, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate handle memory: %s\n", cudaGetErrorString(err));
        put_buffer(ioctl_fd, queue_memory);
        cudaHostUnregister(io_mem);
        return ENOMEM;
    }

    // Allocate PRP list ranges
    //err = cudaHostAlloc(


    // Allocate data buffer
    data_buffer = get_buffer(ioctl_fd, dev, 0x1000 * n_threads);
    if (data_buffer == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        status = ENOMEM;
        goto descriptors;
    }

    // Create IO queue pairs
    if (create_queues(ctrl, queue_memory, queue_descriptors, n_threads, reg_ptr) != 0)
    {
        status = ENOMEM;
        goto data_buffer;
    }

    // Do some work
    status = launch_kernel(ctrl, queue_descriptors, data_buffer, n_threads, ns_id);

    // Clean up and exit
delete_queues:
    // TODO

data_buffer:
    put_buffer(ioctl_fd, data_buffer);

descriptors:
    cudaFreeHost(queue_descriptors);

queue_memory:
    put_buffer(ioctl_fd, queue_memory);

unregister:
    cudaHostUnregister(io_mem);

exit:
    return status;
}


//extern "C" __host__
//int cuda_workload(int ioctl_fd, nvm_controller_t ctrl, int dev, void* reg_ptr, size_t reg_len)
//{
//    cudaError_t err = cudaSetDevice(dev);
//    if (err != cudaSuccess)
//    {
//        fprintf(stderr, "Failed to set CUDA device: %s\n", cudaGetErrorString(err));
//        return EBADF;
//    }
//
//    nvm_queue_t host_sq;
//    nvm_queue_t host_cq;
//    int status = create_queues(ioctl_fd, ctrl, dev, &host_cq, &host_sq);
//    if (status != 0)
//    {
//        fprintf(stderr, "Failed to create queues: %s\n", strerror(status));
//        return status;
//    }
//
//    nvm_queue_t dev_sq;
//    //nvm_queue_t dev_cq;
//
//    err = cudaMalloc(&dev_sq, sizeof(struct nvm_queue));
//    if (err != cudaSuccess)
//    {
//        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
//        return ENOMEM;
//    }
//
////    err = cudaMalloc(&dev_cq, sizeof(struct nvm_queue));
////    if (err != cudaSuccess)
////    {
////        cudaFree(dev_sq);
////        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
////        return ENOMEM;
////    }
//
//    memory_t* host_buffer = get_gpu_buffer(ioctl_fd, dev, sizeof(uint32_t));
//    if (host_buffer == NULL)
//    {
//        cudaFree(dev_sq);
//        //cudaFree(dev_cq);
//        fprintf(stderr, "Failed to allocate buffer\n");
//        return ENOMEM;
//    }
//
//    memory_t* dev_buffer;
//    err = cudaMalloc(&dev_buffer, sizeof(memory_t) + sizeof(uint64_t) * host_buffer->n_addrs);
//    if (err != cudaSuccess)
//    {
//        put_gpu_buffer(ioctl_fd, host_buffer);
//        cudaFree(dev_sq);
//        //cudaFree(dev_cq);
//        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
//        return ENOMEM;
//    }
//
//    cudaHostRegister(reg_ptr, reg_len, cudaHostRegisterIoMemory);
//    
//    void* db;
//    cudaHostGetDevicePointer(&db, reg_ptr, 0);
//    host_sq->db = SQ_DBL(db, host_sq->no, ctrl->dstrd);
//
//    //prepare_read_cmd(host_sq, 1, 512, host_buffer, 0, 1);
//    uint32_t result = 0xcafebabe;
//    //prepare_write_cmd(host_sq, 1, 512, host_buffer, 0, 1);
//    //uint32_t result = 0xdeadbeef;
//
//    cudaMemcpy(dev_sq, host_sq, sizeof(struct nvm_queue), cudaMemcpyHostToDevice);
//    //cudaMemcpy(dev_cq, host_cq, sizeof(struct nvm_queue), cudaMemcpyHostToDevice);
//    cudaMemcpy(dev_buffer, host_buffer, sizeof(memory_t) + sizeof(uint64_t) * host_buffer->n_addrs, cudaMemcpyHostToDevice);
//
//    //cudaMemset(host_buffer->virt_addr, 0xca, sizeof(uint32_t));
//    cudaMemcpy(host_buffer->virt_addr, &result, sizeof(uint32_t), cudaMemcpyHostToDevice);
//
//    // this works on gpu too
//    //sq_submit(host_sq);
//
//    uint32_t* v;
//    cudaMalloc(&v, sizeof(uint32_t));
//
//    do_work<<<1, 1>>>(dev_buffer, dev_sq, v);
//
//    // hack
//    usleep(5000000);
//
//    fprintf(stderr, "Polling...\n");
//    //struct completion* cpl = cq_dequeue_block(host_cq, ctrl);
////    if (cpl != NULL)
////    {
////        fprintf(stderr, "cid=%u sct=%x sc=%x\n", *CPL_CID(cpl), SCT(cpl), SC(cpl));
////    }
//
//    struct completion cpl;
//    memset(&cpl, 0xff, sizeof(cpl));
//    fprintf(stderr, "cid=%u sct=%x sc=%x\n", *CPL_CID(&cpl), SCT(&cpl), SC(&cpl));
//    cudaMemcpy(&cpl, host_cq->page.virt_addr, sizeof(struct completion), cudaMemcpyDeviceToHost);
//    fprintf(stderr, "cid=%u sct=%x sc=%x\n", *CPL_CID(&cpl), SCT(&cpl), SC(&cpl));
//    
//        
//    result = 0xfefefefe;
//    //cudaMemcpy(&result, value, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&result, host_buffer->virt_addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//
//    fprintf(stderr, "%x\n", result);
//
//    cudaMemcpy(&result, v, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    fprintf(stderr, "%x\n", result);
//
//    cudaFree(dev_buffer);
//    put_gpu_buffer(ioctl_fd, host_buffer);
//    cudaFree(dev_sq);
////    cudaFree(dev_cq);
//    return 0;
//}
//

