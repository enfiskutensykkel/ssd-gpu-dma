#include "barrier.h"
#include <thread>
#include <mutex>



Barrier::Barrier(int numThreads)
    : reset(numThreads)
    , arrived(0)
    , flag(0)
{
}


void Barrier::wait()
{
    mtx.lock();

    int localFlag = !flag;

    if (++arrived == reset)
    {
        mtx.unlock();
        arrived = 0;
        flag = localFlag;
    }
    else
    {
        mtx.unlock();

        while (flag != localFlag)
        {
            std::this_thread::yield();
        }
    }
}

