#ifndef __SIMPLE_RDMA_QUEUE_H__
#define __SIMPLE_RDMA_QUEUE_H__

#include <nvm_types.h>
#include <vector>
#include "dma.h"

#ifndef __CUDACC__
#define __host__
#endif

typedef std::vector<nvm_queue_t> QueueList;



__host__
void createQueues(nvm_rpc_t reference, 
                  nvm_ctrl_t controller, 
                  DmaPtr queueMemory, 
                  QueueList& queues);

#endif
