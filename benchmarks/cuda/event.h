#ifndef __BENCHMARK_EVENT_H__
#define __BENCHMARK_EVENT_H__

#include <cuda.h>
#include <string>
#include <stdexcept>


struct Event
{
    cudaEvent_t event;

    Event(cudaStream_t stream = 0)
    {
        auto err = cudaEventCreateWithFlags(&event, cudaEventDefault);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to create event: ") + cudaGetErrorString(err));
        }

        err = cudaEventRecord(event, stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to record event on stream: ") + cudaGetErrorString(err));
        }

    }


    ~Event()
    {
        cudaEventDestroy(event);
    }


    double operator-(const Event& other) const
    {
        float msecs = 0;
        auto err = cudaEventElapsedTime(&msecs, other.event, event);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Could not calculate elapsed time: ") + cudaGetErrorString(err));
        }

        return ((double) msecs) * 1e3;
    }
};


#endif
