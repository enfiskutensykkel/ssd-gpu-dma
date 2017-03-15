#include <cuda.h>
#include "cuda.h"
#include "nvme.h"
#include "nvme_queue.h"
#include "page.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>


static int create_io_queues(nvm_controller_t ctrl, int fd, int dev, size_t num_pairs)
{
    int status;

    // Prepare a number of IO queue pairs
    for (size_t i = 0; i < num_pairs; ++i)
    {
        status = prepare_queue_handles(ctrl);
        if (status != 0)
        {
            fprintf(stderr, "Failed to create queue handles\n");
            return status;
        }

        nvm_queue_t sq = ctrl->queue_handles[ctrl->n_queues - 2];
        status = get_page(&sq->page, fd, dev);
        if (status != 0)
        {
            fprintf(stderr, "Failed to allocate and pin SQ memory\n");
            return status;
        }

        nvm_queue_t cq = ctrl->queue_handles[ctrl->n_queues - 1];
        status = get_page(&cq->page, fd, dev);
        if (status != 0)
        {
            fprintf(stderr, "Failed to allocate and pin CQ memory\n");
            return status;
        }
    }

    // Submit all prepared queues to the controller
    status = create_queues(ctrl);
    if (status != 0)
    {
        fprintf(stderr, "Failed to submit prepared queues\n");
        return status;
    }

    return 0;
}


int start_kernel(int fd, int dev, nvm_controller_t ctrl)
{
    int status;

    status = create_io_queues(ctrl, fd, dev, 1);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create IO queues\n");
        return status;
    }   

    return 0;
}
