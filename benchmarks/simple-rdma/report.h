#ifndef __SIMPLE_RDMA_REPORT_H__
#define __SIMPLE_RDMA_REPORT_H__

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <cstdio>


static inline void report(const std::string& str)
{
    int lineLength = 70;
    std::string padding;
    for (int i = 0; i < lineLength; ++i)
    {
        padding += ".";
    }

    lineLength -= str.length();
    if (lineLength < 0)
    {
        lineLength = 0;
    }

    fprintf(stdout, "%s%*.*s", str.c_str(), lineLength, lineLength, padding.c_str());
    fflush(stdout);
}


static inline void report(const char* str)
{
    report(std::string(str));
}


static inline void report(bool success)
{
    fprintf(stdout, "%-4s\n", success ? "OK" : "FAIL");
    fflush(stdout);
}


static inline void report(cudaError_t err)
{
    report(err == cudaSuccess);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    }
}


static inline void report(int err)
{
    report(err == 0);
    if (err != 0)
    {
        fprintf(stderr, "%s\n", strerror(err));
    }
}


static inline void report(const std::runtime_error& err)
{
    report(false);
    fprintf(stderr, "%s\n", err.what());
}


#endif
