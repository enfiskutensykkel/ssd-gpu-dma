#ifndef __LATENCY_BENCHMARK_BARRIER_H__
#define __LATENCY_BENCHMARK_BARRIER_H__


#include <mutex>


class Barrier
{
    public:
        Barrier(int numThreads);

        void wait();

    private:
        const int reset;
        int arrived;
        int flag;
        std::mutex mtx;
};

#endif
