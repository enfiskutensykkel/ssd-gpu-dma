#define _GNU_SOURCE
#define __DIS_CLUSTER__

#ifndef _REENTRANT
#define _REENTRANT
#endif

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>

#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_admin.h>
#include <nvm_aq.h>
#include <nvm_rpc.h>
#include <nvm_cmd.h>
#include <nvm_queue.h>
#include <nvm_ctrl.h>
#include <sisci_api.h>


static bool _initialized = false;


struct libnvmfio_options
{
    void* pad;
    uint32_t fdid;
    unsigned int reset;
    unsigned int remote_queue;
    unsigned int queue_id;
    unsigned int adapter;
    unsigned int namespace;
};



struct libnvmfio_thread
{
    nvm_ctrl_t* ctrl;   /* Controller handle */
    struct nvm_ctrl_info ctrl_info;
    struct nvm_ns_info ns_info;
    nvm_dma_t* aq_mem;  /* Admin queue memory */
    nvm_aq_ref aq_ref;     /* Admin queue reference */
    nvm_dma_t* cq_mem;   /* Completion queue memory */
    nvm_queue_t cq;     /* Completion queue */
    nvm_dma_t* sq_mem;   /* Submission queue memory */
    nvm_queue_t sq;     /* Submission queue */
    nvm_dma_t* io_mem;   /* Data buffer memory */
    nvm_prp_list_t* prp_lists;
};



static int identify(struct libnvmfio_thread* data)
{
    int err;
    nvm_dma_t* page;

    err = nvm_dis_dma_create(&page, data->ctrl, data->ctrl->page_size, 0);
    if (!nvm_ok(err))
    {
        return err;
    }

    err = nvm_admin_ctrl_info(data->aq_ref, &data->ctrl_info, page->vaddr, page->ioaddrs[0]);
    if (!nvm_ok(err))
    {
        nvm_dma_unmap(page);
        return EIO;
    }

    err = nvm_admin_ns_info(data->aq_ref, &data->ns_info, 1, page->vaddr, page->ioaddrs[0]);
    if (!nvm_ok(err))
    {
        nvm_dma_unmap(page);
        return EIO;
    }

    nvm_dma_unmap(page);
    return 0;
}


static int create_queues(struct libnvmfio_thread* data, unsigned int queue_no, size_t iodepth, unsigned int memhints)
{
    int err;
    size_t i;
    const nvm_ctrl_t* ctrl = data->ctrl;

    ++iodepth;

    data->prp_lists = calloc(iodepth, sizeof(nvm_prp_list_t));
    assert(data->prp_lists);

    err = nvm_dis_dma_create(&data->cq_mem, ctrl, 
            NVM_CQ_PAGES(ctrl, iodepth) * ctrl->page_size, 0);
    if (err != 0)
    {
        free(data->prp_lists);
        return err;
    }

    err = nvm_dis_dma_create(&data->sq_mem, ctrl,
            (NVM_SQ_PAGES(ctrl, iodepth) + iodepth) * ctrl->page_size, memhints);
    if (err != 0)
    {
        nvm_dma_unmap(data->cq_mem);
        free(data->prp_lists);
        return err;
    }

    memset(data->cq_mem->vaddr, 0, data->cq_mem->page_size * data->cq_mem->n_ioaddrs);
    err = nvm_admin_cq_create(data->aq_ref, &data->cq, queue_no, data->cq_mem, 0, iodepth);
    if (err != 0)
    {
        goto unwind;
    }

    memset(data->sq_mem->vaddr, 0, data->sq_mem->page_size * data->sq_mem->n_ioaddrs);
    err = nvm_admin_sq_create(data->aq_ref, &data->sq, &data->cq, queue_no, data->sq_mem, 0, iodepth);
    if (err != 0)
    {
        goto unwind;
    }

    for (i = 0; i < iodepth; ++i)
    {
        data->prp_lists[i] = NVM_PRP_LIST(data->sq_mem, NVM_SQ_PAGES(ctrl, iodepth + i));
    }

    return 0;

unwind:
    nvm_dma_unmap(data->sq_mem);
    nvm_dma_unmap(data->cq_mem);
    free(data->prp_lists);
    return err;
}



static void destroy_queues(struct libnvmfio_thread* data)
{
    nvm_admin_sq_delete(data->aq_ref, &data->sq, &data->cq);
    nvm_dma_unmap(data->sq_mem);
    nvm_admin_cq_delete(data->aq_ref, &data->cq);
    nvm_dma_unmap(data->cq_mem);
    free(data->prp_lists);
}



static int thread_setup(struct thread_data* td)
{
    unsigned int i;
    struct fio_file* f;

    if (!td->o.use_thread)
    {
        /* SPDK checks for this, should we too? */
    }

    if (!_initialized)
    {
        fprintf(stderr, "Failed to initialize SISCI\n");
        return -1;
    }

    for_each_file(td, f, i)
    {
        fprintf(stderr, "%s\n", f->file_name);
    }

    return 0;
}


static enum fio_q_status fio_queue(struct thread_data* td, struct io_u* io_u)
{
    struct libnvmfio_thread* data = td->io_ops_data;
    nvm_queue_t* sq = &data->sq;
    nvm_queue_t* cq = &data->cq;
    const size_t ps = data->ctrl->page_size;
    const size_t bs = data->ns_info.lba_data_size;
    unsigned int idx;
    uint8_t op;
    nvm_cmd_t cmd;
    nvm_cmd_t* ptr;
    nvm_cpl_t* cpl;
    size_t offset;

    offset = ((char*) io_u->buf) - ((char*) data->io_mem->vaddr);

    switch (io_u->ddir)
    {
        case DDIR_READ:
            op = NVM_IO_READ;
            break;

        case DDIR_WRITE:
            op = NVM_IO_WRITE;
            break;

        default:
            assert(false);
            break;
    }

    idx = sq->tail;

    nvm_cmd_header(&cmd, idx, op, data->ns_info.ns_id);

    nvm_cmd_data(&cmd, 1, &data->prp_lists[idx], io_u->xfer_buflen / ps, &data->io_mem->ioaddrs[offset / ps]);
    nvm_cmd_rw_blks(&cmd, io_u->offset / bs, io_u->xfer_buflen / bs);

    ptr = nvm_sq_enqueue(sq);
    if (ptr == NULL)
    {
        return FIO_Q_BUSY;
    }

    *ptr = cmd;
    nvm_sq_submit(sq);

    while ((cpl = nvm_cq_dequeue(cq)) == NULL);
    nvm_sq_update(sq);

    if (!NVM_ERR_OK(cpl))
    {
        fprintf(stderr, "error %s\n", nvm_strerror(NVM_ERR_STATUS(cpl)));
    }

    nvm_cq_update(cq);

    return FIO_Q_COMPLETED;
}


static int iomem_alloc(struct thread_data* td, size_t size)
{
    int err;
    struct libnvmfio_thread* data = td->io_ops_data;

    if (data->io_mem)
    {
        return EINVAL;
    }

    err = nvm_dis_dma_create(&data->io_mem, data->ctrl, size, 0);
    if (!nvm_ok(err))
    {
        return err;
    }

    td->orig_buffer = data->io_mem->vaddr;
    return 0;
}


static void iomem_free(struct thread_data* td)
{
    struct libnvmfio_thread* data = td->io_ops_data;
    nvm_dma_unmap(data->io_mem);
    data->io_mem = NULL;
}


static int fio_open(struct thread_data* td, struct fio_file* f)
{
//    struct libnvmfio_thread* data = td->io_ops_data;
//
//    f->filetype = FIO_TYPE_BLOCK;
//    f->real_file_size = data->ns_info.size * data->ns_info.lba_data_size;
//    fio_file_set_size_known(f);
//
    return 0;
}


static int fio_close(struct thread_data fio_unused *td, struct fio_file fio_unused *f)
{
    return 0;
}


static int fio_invalidate(struct thread_data fio_unused *td, struct fio_file fio_unused *f)
{
    return 0;
}


static int thread_init(struct thread_data* td)
{
    int err;
    struct libnvmfio_options* opts = td->eo;
    struct libnvmfio_thread* data;
    char diskname[41];
    unsigned int flags = 0;
    uint16_t n_cqs, n_sqs;

    if (opts->fdid == 0)
    {
        return EBADF;
    }

    if (opts->queue_id == 0)
    {
        return EINVAL;
    }

    if (td->o.iodepth != 1)
    {
        fprintf(stderr, "Warning: IO depth > 1 is not supported\n");
    }

    data = calloc(1, sizeof(struct libnvmfio_thread));
    if (data == NULL)
    {
        return errno;
    }

    err = nvm_dis_ctrl_init(&data->ctrl, opts->fdid);
    if (!nvm_ok(err))
    {
        free(data);
        return err;
    }

    if (opts->reset)
    {
        err = nvm_dis_dma_create(&data->aq_mem, data->ctrl, data->ctrl->page_size * 2, 0);
        if (!nvm_ok(err))
        {
            nvm_ctrl_free(data->ctrl);
            free(data);
            return err;
        }

        err = nvm_aq_create(&data->aq_ref, data->ctrl, data->aq_mem);
        if (!nvm_ok(err))
        {
            nvm_dma_unmap(data->aq_mem);
            nvm_ctrl_free(data->ctrl);
            free(data);
            return err;
        }

        nvm_admin_set_num_queues(data->aq_ref, opts->queue_id, opts->queue_id);
    }
    else
    {
        err = nvm_dis_rpc_bind(&data->aq_ref, data->ctrl, opts->adapter);
        if (!nvm_ok(err))
        {
            nvm_ctrl_free(data->ctrl);
            free(data);
            return err;
        }
    }

    err = identify(data);
    if (err != 0)
    {
        goto error;
    }

    err = nvm_admin_get_num_queues(data->aq_ref, &n_cqs, &n_sqs);
    if (err != 0)
    {
        goto error;
    }

    if (opts->queue_id > n_sqs)
    {
        err = EINVAL;
        goto error;
    }

    flags = opts->remote_queue ? (SCI_MEMACCESS_DEVICE_READ | SCI_MEMACCESS_HOST_WRITE) : 0;
    err = create_queues(data, opts->queue_id, td->o.iodepth, flags);
    if (err != 0)
    {
        goto error;
    }

    fprintf(stderr, "SQ on node: %u\n", nvm_dis_node_from_dma(data->sq_mem));

    strncpy(diskname, data->ctrl_info.model_no, 40);
    diskname[40] = '\0';
    fprintf(stderr, "Using disk: 0x%x %s\n", opts->fdid, diskname);
    
    td->io_ops_data = data;
    return 0;

error:
    nvm_aq_destroy(data->aq_ref);
    nvm_dma_unmap(data->aq_mem);
    nvm_ctrl_free(data->ctrl);
    free(data);
    return err;
}


static void thread_cleanup(struct thread_data* td)
{
    if (td->io_ops_data)
    {
        struct libnvmfio_thread* data = td->io_ops_data;
        td->io_ops_data = NULL;

        destroy_queues(data);
        nvm_aq_destroy(data->aq_ref);
        nvm_dma_unmap(data->aq_mem);
        nvm_ctrl_free(data->ctrl);

        free(data);
    }
}


static struct fio_option options[] =
{
    { .name = "fdid", .lname = "fdid", .alias = "ctrl", .def = "0", .type = FIO_OPT_ULL, .off1 = offsetof(struct libnvmfio_options, fdid), .help = "Fabric device identifier for NVMe controller", .category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = "reset", .lname = "reset_ctrl", .type = FIO_OPT_STR_SET, .off1 = offsetof(struct libnvmfio_options, reset), .help = "Reset controller and become manager", .category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = "remote_queue", .lname = "remote_queue", .type = FIO_OPT_STR_SET, .off1 = offsetof(struct libnvmfio_options, remote_queue), .help = "Allocate submission queue close to controller", .category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = "queue_no", .lname = "queue_no", .alias = "queue_id", .type = FIO_OPT_ULL, .off1 = offsetof(struct libnvmfio_options, queue_id), .help = "Submission queue identifier",.category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = "adapter", .lname = "adapter", .def = "0", .type = FIO_OPT_ULL, .off1 = offsetof(struct libnvmfio_options, adapter), .help = "DIS adapter number", .category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = "namespace", .lname = "nvm_namespace", .alias = "ns", .def = "1", .type = FIO_OPT_ULL, .off1 = offsetof(struct libnvmfio_options, namespace), .help = "NVM namespace", .category = FIO_OPT_C_ENGINE, .group = FIO_OPT_G_INVALID, },
    { .name = NULL, }
};


struct ioengine_ops engine = 
{
    .name = "libnvm",
    .version = FIO_IOOPS_VERSION,
    .flags = FIO_SYNCIO | FIO_RAWIO | FIO_NOEXTEND | FIO_NODISKUTIL | FIO_MEMALIGN,
    .setup = thread_setup,
    .init = thread_init,
    .queue = fio_queue,
    .cleanup = thread_cleanup,
    .open_file = fio_open,
    .close_file = fio_close,
    .invalidate = fio_invalidate,
    .iomem_alloc = iomem_alloc,
    .iomem_free = iomem_free,
    //.get_file_size
    .option_struct_size = sizeof(struct libnvmfio_options),
    .options = options,
};


static void fio_init libnvm_fio_register(void)
{
    sci_error_t err;

    SCIInitialize(0, &err);
    if (err == SCI_ERR_OK)
    {
        _initialized = true;
        register_ioengine(&engine);
    }
}


static void fio_exit libnvm_fio_unregister(void)
{
    if (_initialized)
    {
        unregister_ioengine(&engine);
        SCITerminate();
        _initialized = false;
    }
}
