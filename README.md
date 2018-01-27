libnvm: A library for building userspace NVM Express drivers
==============================================================================
This library is intended to allow userspace programs to control and manage 
NVM Express (NVMe) disk controllers through an easy-to-use API. The motivation 
is to provide a userspace library your CUDA applications and other programs 
can easily link against, in order to build custom distributed drivers for 
NVMe disk drives.

The library provides simple semantics and functions for mapping userspace
buffers and device memory, providing a simple yet low-level mechanism suitable
for controlling an NVMe disk. By mapping buffers, an NVMe drive is able to 
access these buffers directly (DMA), greatly increasing the IO performance
compared to accessing the drive through normal filesystem abstractions 
provided by the Linux kernel.



Quick start
------------------------------------------------------------------------------
You need a PCIe-attached NVMe disk (rev 1.0 or newer). If the disk contains 
any data, it is important that you back this up before proceeding.

Make sure that the following is installed on your system:
  * CMake 3.1 or newer.
  * GCC version 5.4.0 or newer (compiler must compile C99 and C++11).
  * CUDA 8.0 or newer (optional)
  * Dolphin SISCI API 5.5 or newer (optional)

If you plan on using CUDA, you must use the kernel module or Dolphin SmartIO.
See [Kernel module](#using-kernel-module) or [Dolphin SmartIO](#dolphin-smartio).

### How to build ###
Clone repository and enter project root directory.
```
$ mkdir -p build; cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make libnvme     # builds library
$ make examples    # builds example programs
$ cd module; make  # builds kernel module (optional)
```

### Checking that it works ###
Find out the disk's BDF using `lspci`. For this README, we assume that the
BDF is `05:00.0`.

First unbind the default nvme driver from the disk:
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
```

Then run the identify sample (standing in the build directory). It should
look something like this:
```
$ samples/userspace_identify --ctrl=05:00.0
Resetting controller and setting up admin queues...
------------- Controller information -------------
PCI Vendor ID           : 86 80
PCI Subsystem Vendor ID : 86 80
NVM Express version     : 1.0.0
Controller page size    : 4096
Max queue entries       : 4096
Serial Number           : CVCQ5251008G400AGN
Model Number            : INTEL SSDPEDMW400G4
Firmware revision       : 8EV101H0
Max data transfer size  : 131072
Max outstanding commands: 0
Max number of namespaces: 1
--------------------------------------------------
```


Using kernel module
------------------------------------------------------------------------------
Using the kernel module is not required, unless you plan on using the 
library with CUDA support.
Currently the only version of Linux supported is Linux 4.11.0. Older version
may work, but you probably have to change the call to `get_user_pages()`
and the DMA API.

You should make sure that you use a processor that supports PCIe peer-to-peer,
for example Intel Xeon, and that you have a GPU with GPUDirect support (Quadro
or Tesla workstation). For best support, check that your GPU is Pascal 
architecture or newer. Currently, IOMMU support is broken, so disable the
IOMMU. This is done by removing `iommu=on` and `intel_iommu=on` from the `CMDLINE`
variable in `/etc/default/grub`. 

Loading and unloading the driver is done as follows:
```
$ cd build/module
$ make
$ make load     # will insert the kernel module
$ make unload   # unloads the kernel module
```

You want to unload the default nvme driver for the NVMe disk, and bind 
the disnvm driver to it (replace `05:00.0` with disk BDF):
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
$ echo -n "0000:05:00.0" > /sys/bus/pci/drivers/disnvm/bind
```

After doing this, the file `/dev/disnvm0` should show up, representing the
disk's BAR0.
