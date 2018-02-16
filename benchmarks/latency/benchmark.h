#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__


#include <vector>
#include <chrono>
#include <cstdint>


typedef std::chrono::duration<double, std::micro> mtime;



struct Time
{
    uint16_t    commands;
    size_t      blocks;
    mtime       time;
    
    Time(uint16_t commands, size_t blocks, mtime time)
        : commands(commands), blocks(blocks), time(time) {}
};


typedef std::vector<Time> Times;


#endif
