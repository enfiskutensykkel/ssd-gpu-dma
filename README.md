libnvme - Userspace library for accessing NVM Express drives
==============================================================================

This project is intended to allow userspace programs to control and manage 
NVM Express (NVMe) drives through an easy-to-use API. The motivation for this
is to create an easy-to-use library your program can link against in order to
use one or more PCIe-based SSDs as generic persistent storage memory with
block read/write semantics.

By providing a simple yet low-level interface for NVMe drives, the API is also
able to provide efficient IO semantics. Among the provided sample programs,
we demonstrate that it is possible to do IO operations over PCIe in a more
efficient manner than through the normal filesystem abstraction provided by 
the Linux kernel. We demonstrate that that mapping IO transfers directly on
to device memory, rather than bouncing through RAM buffers, is a lot more
efficient.
