#ifndef __SIMPLE_RDMA_BENCHMARK_H__
#define __SIMPLE_RDMA_BENCHMARK_H__

#include "settings.h"
#include "dma.h"
#include "transfer.h"
#include "queue.h"


double benchmark(QueueList& queues, const TransferList& transfers, DmaPtr hostBuffer, DmaPtr deviceBuffer);

double benchmark(QueueList& queues, const TransferList& transfers, DmaPtr deviceBuffer);

#endif
