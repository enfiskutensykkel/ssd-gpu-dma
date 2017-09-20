cmake_minimum_required (VERSION 3.1)
project (disnvme)

make_sample (userspace_identify userspace_identify "no_module.c")

make_sample (module_identify module_identify "with_module.c")
set_module (module_identify)
