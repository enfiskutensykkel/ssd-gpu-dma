libdisnvme = Dolphin + Nvidia CUDA + NVM Express
==============================================================================
Build custom userspace NVM Express drivers with CUDA support.

This library is intended to allow userspace programs to control and manage 
NVM Express (NVMe) disk controllers through an easy-to-use API. The motivation 
is to provide a userspace library your CUDA applications and other programs 
can easily link against, in order to build custom distributed drivers for 
NVMe disk drives.

The library provides simple semantics and functions for mapping userspace
buffers and device memory, providing a simple yet low-level mechanism suitable
for controlling an NVMe drive. By mapping buffers, an NVMe drive is able to 
access these buffers directly (DMA), greatly increasing the IO performance
compared to accessing the drive through normal filesystem abstractions 
provided by the Linux kernel.

