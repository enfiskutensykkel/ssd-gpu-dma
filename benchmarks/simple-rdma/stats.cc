#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <nvm_types.h>
#include "settings.h"
#include "stats.h"


void showStatistics(const Settings& settings, const std::string& title, const std::vector<uint64_t>& times)
{
    //double totalSize = times.size() * settings.numBlocks * settings.blockSize;
    //double totalTime = 0;

    auto printline = [](char c) {
        for (size_t i = 0; i < 70; ++i)
        {
            fputc(c, stdout);
        }
        fprintf(stdout, "\n");
    };

    double minBw = std::numeric_limits<double>::max();
    double maxBw = std::numeric_limits<double>::min();

    double avgBw = 0;

    for (const uint64_t time: times)
    {
        //totalTime += ((double) time) / 1e6;
        //minTime = std::min(time, minTime);
        //maxTime = std::max(time, maxTime);

        double bw = ((settings.numBlocks * settings.blockSize) >> 20) / (((double) time) / 1e6);
        minBw = std::min(bw, minBw);
        maxBw = std::max(bw, maxBw);
        avgBw += bw;
    }
    avgBw /= times.size();

    printline('=');
    fprintf(stdout, " %s\n", title.c_str());
    fprintf(stdout, " Queue depth: %zu x %zu, Transfer size: %zu KiB, Num blocks: %zu \n", 
            settings.numQueues, 0x1000 / sizeof(nvm_cmd_t), settings.chunkSize >> 10, settings.numBlocks);
    printline('-');
    fprintf(stdout, " %-16s\t%-16s\t%-16s\n", "Minimum", "Mean", "Maximum");
    fprintf(stdout, " %10.3f MiB/s\t%10.3f MiB/s\t%10.3f MiB/s\n", minBw, avgBw, maxBw);
    printline('=');
}

