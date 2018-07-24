#ifndef __LATENCY_BENCHMARK_BENCHMARK_H__
#define __LATENCY_BENCHMARK_BENCHMARK_H__


#include <chrono>
#include <vector>
#include <memory>
#include <cstdint>
#include "transfer.h"
#include "settings.h"


/*
 * Create a convenience type for representing microseconds.
 */
typedef std::chrono::duration<double, std::micro> usec_t;



/*
 * Record time for a number of commands, that is, submitting a bunch of
 * commands and waiting for their completions. On a per-command basis,
 * this would be the equivalent of number of IO operations per second.
 */
struct Event
{
    size_t      commands;   // Number of commands
    size_t      blocks;     // Number of blocks
    usec_t      time;       // Number of microseconds

    Event(size_t ncmds, size_t nblks, usec_t usecs)
        : commands(ncmds), blocks(nblks), time(usecs)
    { }


    /*
     * Calcluate number of IO operations per second (IOPS).
     * This will only be an estimate, unless commands == 1.
     */
    double estimateIops() const
    {
        return 1e6 / averageLatencyPerCommand();
    }


    /*
     * Estimate number of IO operations per second (IOPS) 
     * adjusted for increased transfer sizes.
     */
    double adjustedIops(size_t blocksPerPrp = 1) const
    {
        return 1e6 / (averageLatencyPerBlock() * blocksPerPrp);
    }


    /*
     * Calculate average number of microseconds per block.
     */
    double averageLatencyPerBlock() const
    {
        return time.count() / blocks;
    }


    /*
     * Calculate average number of microseconds per command.
     */
    double averageLatencyPerCommand() const
    {
        return (time.count() / commands);
    }


    size_t transferSize(size_t blockSize) const
    {
        return blocks * blockSize;
    }


    double bandwidth(size_t blockSize) const
    {
        return (blocks * blockSize) / time.count();
    }
};



/*
 * Convenience types for a series of recorded events.
 */
typedef std::vector<Event> EventList;
typedef std::shared_ptr<EventList> EventListPtr;
typedef std::map<uint16_t, EventListPtr> EventMap;



void benchmark(EventMap& times, const TransferMap& transfers, const Settings& settings, bool write);

#endif /* __LATENCY_BENCHMARK_BENCHMARK_H__ */
