#ifndef __BENCHMARK_EVENT_H__
#define __BENCHMARK_EVENT_H__

#include <cuda.h>
#include <string>
#include <stdexcept>


struct Event
{
    cudaEvent_t event;

    Event()
    {
        cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDefault);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to create event: ") + cudaGetErrorString(err));
        }
    }

    void record(cudaStream_t stream)
    {
        cudaError_t err = cudaEventRecord(event, stream);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to record event on stream: ") + cudaGetErrorString(err));
        }
    }

    ~Event()
    {
        cudaEventDestroy(event);
    }
};


#endif
