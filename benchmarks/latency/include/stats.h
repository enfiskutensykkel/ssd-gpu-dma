#ifndef __LATENCY_BENCHMARK_STATISTICS_H__
#define __LATENCY_BENCHMARK_STATISTICS_H__

#include <cstddef>
#include "ctrl.h"
#include "queue.h"
#include "benchmark.h"
#include "transfer.h"


void printPercentiles(const EventMap& events, const QueueMap& transfers, bool latency, bool write);


void printStatistics(const EventMap& readEvents, const EventMap& writeEvents, const TransferMap& transfers, const Settings& settings);


#endif /* __LATENCY_BENCHMARK_STATISTICS_H__ */
