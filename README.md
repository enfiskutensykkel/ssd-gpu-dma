Controlling an NVMe from a CUDA kernel
======================================

NVMe
--------------------------------------
[NVM Express](http://www.nvmexpress.org/wp-content/uploads/NVM_Express_1_2_1_Gold_20160603.pdf)

For simplicity is every queue a page large

sysfs
--------------------------------------
[Using sysfs to access PCI device resources](https://www.kernel.org/doc/Documentation/filesystems/sysfs-pci.txt)

Nvidia CUDA
--------------------------------------
  - [GPUDirect RDMA](http://docs.nvidia.com/cuda/gpudirect-rdma/#axzz4a6tqNDAe)

  - GPUDirect Async

Loadable Kernel Module
--------------------------------------
Linux 4.4.0 family due to changing signature of `get_user_pages`

Limitations
--------------------------------------
IOMMU stuff
pages allocated in userspace are not contiguous (why everything is operating on pages)


Future optimisations
--------------------------------------
  - Use controller onboard memory (optional in NVMe) to host queues and push data using GPUDirect Async
