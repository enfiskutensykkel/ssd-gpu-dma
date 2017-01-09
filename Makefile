TARGET	:= ssd_dma
OBJECTS := src/main.o
DISHOME := /opt/DIS
RELEASE	:= $(shell uname -r)

obj-m := $(TARGET).o
$(TARGET)-objs := $(OBJECTS)
KDIR ?= /lib/modules/$(RELEASE)/build

ifeq ($(KERNELRELEASE),)
	CC 	:= gcc
	CFLAGS	:= -Wall -Wextra -pedantic -D_REENTRANT -Wno-unused-parameter
	INCLUDE	:= -I$(DISHOME)/include -I$(DISHOME)/include/dis -Iinclude
	LDLIBS	:= -lsisci -lpthread
	LDFLAGS	:= -L$(DISHOME)/lib64
endif

.PHONY: default reload unload load

default: modules userspace

userspace: src/userspace.c
	$(CC) -std=gnu99 -o $@ $(CFLAGS) $(INCLUDE) $(LDFLAGS) $^ $(LDLIBS)


reload: unload load

unload:
	-rmmod $(TARGET).ko

load:
	insmod $(TARGET).ko

clean:
	-$(RM) userspace
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	
%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@

