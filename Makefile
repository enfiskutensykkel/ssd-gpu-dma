PROJECT 		:= cunvme
DEFINES			:= -DMAX_RAM_PAGES=64 -DMAX_GPU_PAGES=8

# These should be overrided from command line
# e.g. make CUDA_PATH=/usr/local/cuda-8.0 NV_DRIVER_PATH=/opt/nvidia-367.48
KERNEL_RELEASE 	:= $(shell uname -r)
CUDA_PATH 	:= /usr/local/cuda
NV_DRIVER_PATH	:= /usr/src/nvidia-367.48

# If we're building userspace, use these variables:
CC 		:= $(CUDA_PATH)/bin/nvcc
CCBIN		:= /usr/bin/gcc
CFLAGS		:= -Wall -Wextra -Werror=implicit-function-declaration -O2
INCLUDE		:= -I$(CUDA_PATH)/include -Iinclude  -Iuserspace
DEFINES		+= -DCUNVME_PATH='"/proc/$(PROJECT)"'
LDLIBS		:= -lcuda -lc 
LDFLAGS		:= -L$(CUDA_PATH)/lib64
SOURCE_FILES	:= $(shell find userspace/ -type f -name "*.cu") $(shell find userspace/ -type f -name "*.c")
HEADER_FILES	:= $(shell find userspace/ -type f -name "*.h")
OBJECT_FILES	:= $(SOURCE_FILES:%=%.o)


## Set make targets
.PHONY: default reload unload load clean


default: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECT_FILES)

$(PROJECT): $(OBJECT_FILES)
	$(CCBIN) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	#$(CC) -ccbin $(CCBIN) -o $@ $^ $(LDFLAGS) $(LDLIBS)


# How to compile C files
userspace/%.c.o: userspace/%.c $(HDR_FILES)
	$(CCBIN) -std=gnu11 $(CFLAGS) $(DEFINES) $(INCLUDE) -o $@ $< -c

# How to compile CUDA files
userspace/%.cu.o: userspace/%.cu $(HDR_FILES)
	$(CC) -std=c++11 -Xcompiler "-D__CORRECT_ISO_CPP11_MATH_H_PROTO $(CFLAGS) $(DEFINES) -D__CORRECT_ISO_CPP11_MATH_H_PROTO" $(INCLUDE) -o $@ $< -c -O3 -dc --gpu-architecture=compute_50 --gpu-code=compute_50,sm_50


# Hack to mix userspace and kernel module stuff together in same makefile
%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@
