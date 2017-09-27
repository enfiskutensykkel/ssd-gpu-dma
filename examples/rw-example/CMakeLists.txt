cmake_minimum_required (VERSION 3.1)
project (disnvme)

make_sample (rw-example rw "rw.c;transfer.c")
set_multithread (rw-example)
