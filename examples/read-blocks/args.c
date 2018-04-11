#include "args.h"
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <fcntl.h>
#include <limits.h>
#include <errno.h>
#include <string.h>



static struct option opts[] = {
    { .name = "help", .has_arg = no_argument, .flag = NULL, .val = 'h' },
    { .name = "ctrl", .has_arg = required_argument, .flag = NULL, .val = 'c' },
#ifdef __DIS_CLUSTER__
    { .name = "adapter", .has_arg = required_argument, .flag = NULL, .val = 'a' },
    { .name = "segment-id", .has_arg = required_argument, .flag = NULL, .val = 1 },
    { .name = "id-offset", .has_arg = required_argument, .flag = NULL, .val = 1 },
#endif
    { .name = "namespace", .has_arg = required_argument, .flag = NULL, .val = 'n' },
    { .name = "ns", .has_arg = required_argument, .flag = NULL, .val = 'n' },
    { .name = "blocks", .has_arg = required_argument, .flag = NULL, .val = 'b' },
    { .name = "offset", .has_arg = required_argument, .flag = NULL, .val = 'o' },
    { .name = "output", .has_arg = required_argument, .flag = NULL, .val = 0 },
    { .name = "ascii", .has_arg = no_argument, .flag = NULL, .val = 2 },
    { .name = "identify", .has_arg = no_argument, .flag = NULL, .val = 3 },
    { .name = "chunk", .has_arg = required_argument, .flag = NULL, .val = 's' },
    { .name = "write", .has_arg = required_argument, .flag = NULL, .val = 'w' },
    { .name = NULL, .has_arg = no_argument, .flag = NULL, .val = 0 }
};



static void show_usage(const char* name)
{
#ifdef __DIS_CLUSTER__
    fprintf(stderr, "Usage: %s --ctrl <device id> --blocks <count> [--offset <count>] [--ns <id>] [--ascii | --output <path>]\n", name);
#else
    fprintf(stderr, "Usage: %s --ctrl <path> --blocks <count> [--offset <count>] [--ns <id>] [--ascii | --output <path>]\n", name);
#endif
}



static void show_help(const char* name)
{
    show_usage(name);

    fprintf(stderr, ""
#ifdef __DIS_CLUSTER__
            "    --ctrl         <id>      Specify controller's device identifier.\n"
            "    --adapter      <no>      Specify local DIS adapter.\n"
#else
            "    --ctrl         <path>    Specify path to controller.\n"
#endif
            "    --chunk        <count>   Limit reads to a number of blocks at the time.\n"
            "    --blocks       <count>   Read specified number of blocks from disk.\n"
            "    --offset       <count>   Start reading at specified block (default 0).\n"
            "    --namespace    <id>      Namespace identifier (default 1).\n"
            "    --ascii                  Show output of ASCII characters as text.\n"
            "    --output       <path>    Dump to file rather than stdout.\n"
            "    --write        <path>    Read file and write to disk before reading back.\n"
            "    --identify               Show IDENTIFY CONTROLLER structure.\n"
           );
}



void parse_options(int argc, char** argv, struct options* args)
{
#ifdef __DIS_CLUSTER__
    const char* argstr = ":hc:a:n:b:o:s:w:";
#else
    const char* argstr = ":hc:b:n:o:s:w:";
#endif

    int opt;
    int idx;
    char* endptr;

#ifdef __DIS_CLUSTER__
    args->controller_id = 0;
    args->adapter = 0;
    args->segment_id = 0xdeadbeef;
    args->chunk_size = (64UL << 20) / 512;
#else
    args->controller_path = NULL;
    args->chunk_size = 0;
#endif
    args->namespace_id = 1;
    args->num_blocks = 0;
    args->offset = 0;
    args->output = NULL;
    args->input = NULL;
    args->ascii = false;
    args->identify = false;

    while ((opt = getopt_long(argc, argv, argstr, opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?':
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                exit('?');

            case ':':
                fprintf(stderr, "Missing argument for option %s\n", argv[optind - 1]);
                exit('?');

            case 'h':
                show_help(argv[0]);
                exit('?');

            case 0:
                if (args->ascii)
                {
                    fprintf(stderr, "Output file is set, ignoring option --ascii\n");
                    args->ascii = false;
                }

                args->output = fopen(optarg, "wb");
                if (args->output == NULL)
                {
                    fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
                    exit(1);
                }
                break;

            case 'w':
                args->input = fopen(optarg, "rb");
                if (args->input == NULL)
                {
                    fprintf(stderr, "Failed to open input file: %s\n", strerror(errno));
                    exit(1);
                }
                break;


            case 3:
                args->identify = true;
                break;

#ifdef __DIS_CLUSTER__
            case 'c':
                args->controller_id = strtoul(optarg, &endptr, 16);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid controller id: `%s'\n", optarg);
                    exit(1);
                }

                if (args->controller_id == 0)
                {
                    fprintf(stderr, "Controller id can not be 0!\n");
                    exit(1);
                }
                break;

            case 'a':
                args->adapter = strtoul(optarg, &endptr, 10);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid adapter number: `%s'\n", optarg);
                    exit(1);
                }

                if (args->adapter >= 4)
                {
                    fprintf(stderr, "Adapter number is too large!\n");
                    exit(1);
                }
                break;

            case 1:
                args->segment_id = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid segment id: `%s'\n", optarg);
                    exit(1);
                }
                break;
#else
            case 'c':
                args->controller_path = optarg;
                break;
#endif

            case 'n':
                args->namespace_id = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0' || args->namespace_id == 0xffffffff)
                {
                    fprintf(stderr, "Invalid namespace identifier: `%s'\n", optarg);
                    exit(2);
                }
                break;

            case 2:
                if (args->output == NULL)
                {
                    args->ascii = true;
                }
                else
                {
                    fprintf(stderr, "Output file is set, ignoring option %s\n", argv[optind - 1]);
                }
                break;

            case 'o':
                args->offset = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }
                break;

            case 'b':
                args->num_blocks = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }

                if (args->num_blocks == 0)
                {
                    fprintf(stderr, "Number of blocks can not be 0!\n");
                    exit(2);
                }
                break;

            case 's':
                args->chunk_size = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }
                break;
        }
    }

#ifdef __DIS_CLUSTER__
    if (args->controller_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        show_usage(argv[0]);
        exit(1);
    }
#else
    if (args->controller_path == NULL)
    {
        fprintf(stderr, "No controller specified!\n");
        show_usage(argv[0]);
        exit(1);
    }
#endif

    if (args->num_blocks == 0)
    {
        fprintf(stderr, "Block count is not specified!\n");
        show_usage(argv[0]);
        exit(2);
    }

    if (args->chunk_size == 0)
    {
        args->chunk_size = args->num_blocks;
    }
}

