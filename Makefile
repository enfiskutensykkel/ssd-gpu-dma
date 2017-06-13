PROJECT 		:= cunvme
DEFINES			:= 

# If we're building userspace, use these variables:
CC		:= /usr/bin/gcc
CFLAGS		:= -Wall -Wextra -Werror=implicit-function-declaration -O2
INCLUDE		:= -Iinclude  -Iuserspace -I/opt/DIS/include -I/opt/DIS/include/dis
DEFINES		+= -DCUNVME_PATH='"/proc/$(PROJECT)"'
LDLIBS		:= -lsisci
LDFLAGS		:= -L/opt/DIS/lib64
SOURCE_FILES	:= $(shell find userspace/ -type f -name "*.cu") $(shell find userspace/ -type f -name "*.c")
HEADER_FILES	:= $(shell find userspace/ -type f -name "*.h")
OBJECT_FILES	:= $(SOURCE_FILES:%=%.o)


## Set make targets
.PHONY: default reload unload load clean


default: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECT_FILES)

$(PROJECT): $(OBJECT_FILES)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

# How to compile C files
userspace/%.c.o: userspace/%.c $(HDR_FILES)
	$(CC) -std=gnu11 $(CFLAGS) $(DEFINES) $(INCLUDE) -o $@ $< -c

# Hack to mix userspace and kernel module stuff together in same makefile
%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@
