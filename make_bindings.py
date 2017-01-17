#!/usr/bin/env python

import re
import os
import sys

functions = [
        'iod_list',
        'nvme_free_iod',
        'nvme_setup_prps'
        ]

structs = [
        'nvme_ctrl',
        'nvme_ns',
        'nvme_dev',
        'nvme_iod'
        ]


def parse_body(string):
    at_least_once = False
    bracket_level = 0
    body = ""

    while bracket_level > 0 or not at_least_once:
        c = string[0]
        string = string[1:]

        if c == '{':
            at_least_once = True
            bracket_level += 1

        elif c == '}':
            bracket_level -= 1

        body += c

    return body


def parse_functions(path, functions, function_bodies):
    data = open(path).read()

    func_decl_pattern = re.compile(r'([#A-Za-z0-9_\*\s]+)\([^\)]+\)\n\{', re.MULTILINE | re.DOTALL)
    func_name_pattern = re.compile(r'([A-Za-z0-9_]+)\s*\(')

    while True:
        hit = func_decl_pattern.search(data)
        if hit is None:
            break

        decl_line = hit.group(0)
        name = func_name_pattern.search(decl_line).group(1)

        if name in functions:
            function_bodies[name] = "\n".join([line for line in parse_body(data[hit.start():]).split('\n') if len(line) < 1 or line[0] != '#'])

        data = data[hit.end():]


def parse_structs(path, structs, struct_bodies):
    data = open(path).read()

    struct_decl_pattern = re.compile(r'struct ([A-Za-z0-9_]+)\s+\{', re.MULTILINE)

    while True:
        hit = struct_decl_pattern.search(data)
        if hit is None:
            break

        struct = hit.group(1)

        if struct in structs:
            struct_bodies[struct] = "\n".join([line for line in parse_body(data[hit.start():]).split('\n') if len(line) < 1 or line[0] != '#'])
            struct_bodies[struct] += ";\n\n"

        data = data[hit.end():]


def parse_includes(path):
    return re.findall(r'#include\s+<[^>]+>', open(path).read())


if len(sys.argv) != 2:
    print >> sys.stderr, "Usage: %s <folder>" % sys.argv[0]
    sys.exit(1)

release = sys.argv[1]
includes = []
func_data = {}
struct_data = {}

for filename in os.listdir(release):
    if filename.split(".")[1] in ['c', 'h']:
        path = os.path.join(release, filename)

        parse_functions(path, functions, func_data)

        parse_structs(path, structs, struct_data)

        for include in parse_includes(path):
            if not include in includes:
                includes.append(include)


with open('include/bind.h', 'w') as f:
    f.write("#ifndef __SSD_DMA_BINDINGS_H__\n")
    f.write("#define __SSD_DMA_BINDINGS_H__\n\n")
    f.write("/* Generated stub, might not work for some kernel versions */\n")
    f.write("\n".join(includes))
    f.write("\n\n\n")

    for struct in structs:
        f.write(struct_data[struct])
        f.write("\n")

    for func in functions:
        f.write(func_data[func])
        f.write("\n")

    f.write("\n\n#endif /* __SSD_DMA_BINDINGS_H__ */\n")
