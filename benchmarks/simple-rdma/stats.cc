#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstdio>
#include "settings.h"
#include "stats.h"


void showStatistics(const Settings& settings, const std::string& title, const std::vector<uint64_t>& times)
{
    double totalSize = times.size() * settings.numBlocks * settings.blockSize;
    double totalTime = 0;

    auto printline = [](char c) {
        for (size_t i = 0; i < 80; ++i)
        {
            fputc(c, stdout);
        }
        fprintf(stdout, "\n");
    };

    uint64_t minTime = std::numeric_limits<uint64_t>::max();
    uint64_t maxTime = std::numeric_limits<uint64_t>::min();

    double minBw = std::numeric_limits<double>::max();
    double maxBw = std::numeric_limits<double>::min();

    double avgBw = 0;

    for (const uint64_t time: times)
    {
        totalTime += time;
        minTime = std::min(time, minTime);
        maxTime = std::max(time, maxTime);

        double bw = (settings.numBlocks * settings.blockSize) / ((double) time);
        minBw = std::min(bw, minBw);
        maxBw = std::max(bw, maxBw);
        avgBw += bw;
    }
    avgBw /= times.size();

    fprintf(stdout, "%lu\n", times[0]);

    printline('=');
    fprintf(stdout, "%s\n", title.c_str());
    printline('-');
    fprintf(stdout, "%10.3f MiB/s\n", totalSize / totalTime);
    fprintf(stdout, "%10.3f MiB/s\t%10.3f MiB/s\t%10.3f MiB/s\n", minBw, avgBw, maxBw);
    fprintf(stdout, "%10lu µs   \t%10lu µs   \t%10lu µs\n", minTime, (uint64_t) totalTime / times.size(), maxTime);
    printline('=');
}

