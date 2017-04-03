PROJECT 		:= cunvme
DEFINES			:= -DMAX_RAM_PAGES=64 -DMAX_GPU_PAGES=8

# These should be overrided from command line
# e.g. make CUDA_PATH=/usr/local/cuda-8.0 NV_DRIVER_PATH=/opt/nvidia-367.48
KERNEL_RELEASE 	:= $(shell uname -r)
CUDA_PATH 		:= /usr/local/cuda
NV_DRIVER_PATH	:= /usr/src/nvidia-367.48

# Kernel module make variables
obj-m 			:= $(PROJECT).o
$(PROJECT)-objs	:= module/$(PROJECT).o
ccflags-y		+= -I$(PWD)/include -DCUNVME_FILENAME='"$(PROJECT)"' -DCUNVME_VERSION='"0.1"' -I$(NV_DRIVER_PATH) 
KDIR			?= /lib/modules/$(KERNEL_RELEASE)/build
KBUILD_EXTRA_SYMBOLS := $(NV_DRIVER_PATH)/Module.symvers

# If we're building userspace, use these variables:
ifeq ($(KERNELRELEASE),)
CC 				:= $(CUDA_PATH)/bin/nvcc
CCBIN			:= /usr/bin/gcc
CFLAGS			:= -Wall -Wextra -Werror=implicit-function-declaration -O2
INCLUDE			:= -I$(CUDA_PATH)/include -Iinclude  -Iuserspace
DEFINES			+= -DCUNVME_PATH='"/proc/$(PROJECT)"'
LDLIBS			:= -lcuda -lc -lpthread
LDFLAGS			:= -L$(CUDA_PATH)/lib64
SOURCE_FILES	:= $(shell find userspace/ -type f -name "*.cu") $(shell find userspace/ -type f -name "*.c")
HEADER_FILES	:= $(shell find userspace/ -type f -name "*.h")
OBJECT_FILES	:= $(SOURCE_FILES:%=%.o)
endif


## Set make targets
.PHONY: default reload unload load clean


default: modules $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECT_FILES)
	$(MAKE) -C $(KDIR) M=$(PWD) clean

$(PROJECT): $(OBJECT_FILES)
	$(CC) -ccbin $(CCBIN) -o $@ $^ $(LDFLAGS) $(LDLIBS)

reload: unload load

unload:
	-rmmod $(PROJECT).ko

load:
	insmod $(PROJECT).ko num_page_handles=128 max_ram_pages=64 max_gpu_pages=8

# How to compile C files
userspace/%.c.o: userspace/%.c $(HDR_FILES)
	$(CCBIN) -std=gnu11 $(CFLAGS) $(DEFINES) $(INCLUDE) -o $@ $< -c

# How to compile CUDA files
userspace/%.cu.o: userspace/%.cu $(HDR_FILES)
	$(CC) -std=c++11 -ccbin $(CCBIN) -Xcompiler "$(CFLAGS) $(DEFINES)" $(INCLUDE) -o $@ $< -c -O2 -dc --gpu-architecture=compute_50 --gpu-code=compute_50,sm_50 

# Hack to mix userspace and kernel module stuff together in same makefile
%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@
