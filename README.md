Controlling an NVMe from a CUDA kernel
======================================

NVMe
--------------------------------------
[http://www.nvmexpress.org/wp-content/uploads/NVM_Express_1_2_1_Gold_20160603.pdf](NVM Express)

For simplicity is every queue a page large

sysfs
--------------------------------------
[https://www.kernel.org/doc/Documentation/filesystems/sysfs-pci.txt](Using sysfs to access PCI device resources)

Nvidia CUDA
--------------------------------------
  - [http://docs.nvidia.com/cuda/gpudirect-rdma/#axzz4a6tqNDAe](GPUDirect RDMA)

  - GPUDirect Async

Loadable Kernel Module
--------------------------------------

Limitations
--------------------------------------
IOMMU stuff

Future optimisations
--------------------------------------
  - Use controller onboard memory (optional in NVMe) to host queues and push data using GPUDirect Async
