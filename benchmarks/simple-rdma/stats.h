#ifndef __SIMPLE_RDMA_STATS_H__
#define __SIMPLE_RDMA_STATS_H__

#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include "settings.h"


void showStatistics(const Settings& settings, 
                    const std::string& title, 
                    const std::vector<uint64_t>& times);

void printStatistics(FILE* fp,
                     const Settings& settings,
                     const std::string& title,
                     const std::vector<uint64_t>& times);

#endif
