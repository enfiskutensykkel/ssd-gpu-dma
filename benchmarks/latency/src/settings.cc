#include "settings.h"
#include "gpu.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <getopt.h>
#include <cstdint>
#include <cstdlib>

using std::string;
using std::vector;
using error = std::runtime_error;


template <typename T>
static bool tryNumber(T& number, const string& str, int base = 0)
{
    char* end = nullptr;
    
    number = strtoul(str.c_str(), &end, base);
    if (end == nullptr || *end != '\0')
    {
        return false;
    }

    return true;
}



static void split(vector<string>& words, const string& str, char delim)
{
    for (size_t pos = 0, end = 0; pos != str.npos; pos = end)
    {
        if (str[pos] == delim)
        {
            ++pos;
        }

        end = str.find(delim, pos);
        words.emplace_back(str, pos, end - pos);
    }
}



#ifdef __DIS_CLUSTER__
static bool tryDeviceId(const string& str, uint64_t& fdid, uint32_t& adapter, bool adapterRequired = false)
{
    vector<string> words;
    split(words, str, ':');

    if (words.empty() || words.size() > 2 || (adapterRequired && words.size() != 2))
    {
        return false;
    }

    if (!tryNumber<uint64_t>(fdid, words[0], 16))
    {
        return false;
    }

    adapter = 0;
    if (words.size() < 2 || words[1].empty())
    {
        return true;
    }

    if (!tryNumber<uint32_t>(adapter, words[1], 0))
    {
        return false;
    }

    return true;
}
#endif



static uint64_t parseNumber(const string& str, int base = 0)
{
    uint64_t number = 0;
    
    if (tryNumber<uint64_t>(number, str, base))
    {
        return number;
    }

    throw error("`" + str + "' is not a number");
}



QueueParam::QueueParam()
{
    no = 0;
    depth = 0;
    pages = 0;
    location = QueueLocation::LOCAL;
    fdid = 0;
    adapter = 0;
}



QueueParam::QueueParam(const string& arg)
{
    no = 0;
    depth = 0;
    pages = 0;
    location = QueueLocation::LOCAL;
    fdid = 0;
    adapter = 0;

    // Split string on ',' and then on '='
    vector<string> words;
    split(words, arg, ',');

    for (const string& w: words)
    {
        vector<string> param;
        split(param, w, '=');

        if (param.empty())
        {
            continue;
        }
        
        if (param.size() != 2)
        {
            throw error("`" + arg + "' is a malformed queue specification string, `" + w + "' is not key=value format");
        }

        if (param[0] == "id" || param[0] == "no" || param[0] == "number")
        {
            if (!tryNumber(no, param[1], 0))
            {
                throw error("`" + param[1] + "' is an invalid queue identifier");
            }
        }
        else if (param[0] == "d" || param[0] == "c" || param[0] == "cmds" || param[0] == "depth" || param[0] == "commands")
        {
            if (!tryNumber(depth, param[1], 0))
            {
                throw error("`" + param[1] + "' is an invalid number of commands (queue depth)");
            }

            if (depth > 64)
            {
                throw error("Number of commands (queue depth) must be between 0 and 64");
            }
        }
        else if (param[0] == "p" || param[0] == "pages" || param[0] == "prps" || param[0] == "chunk")
        {
            if (!tryNumber(pages, param[1], 0))
            {
                throw error("`" + param[1] + "' is an invalid maximum queue PRP count");
            }
        }
        else if (param[0] == "l" || param[0] == "h" || param[0] == "loc" || param[0] == "location" || param[0] == "host")
        {
            if (param[1] == "local" || param[1] == "host" || param[1] == "ram")
            {
                location = QueueLocation::LOCAL;
            }
#ifdef __DIS_CLUSTER__
            else if (param[1] == "remote" || param[1] == "target")
            {
                location = QueueLocation::REMOTE;
            }
            else if (tryDeviceId(param[1], fdid, adapter))
            {
                location = QueueLocation::GPU;
            }
#endif
            else
            {
                throw error("`" + param[1] + "' is an invalid queue location");
            }
        }
        else
        {
            throw error("`" + param[0] + "' is an unknown queue specification key");
        }
    }

    if (no == 0)
    {
        throw error("Queue identifier is not specified");
    }
}



static void setController(CtrlParam& ctrl, const string& arg)
{
    ctrl.path.clear();
    ctrl.fdid = 0;
    ctrl.adapter = 0;

    ctrl.path = arg;

#ifdef __DIS_CLUSTER__
    if (tryDeviceId(arg, ctrl.fdid, ctrl.adapter))
    {
        ctrl.path.clear();
    }
#endif
}



static void setGpu(GpuParam& gpu, const string& arg)
{
    gpu.device = -1;
    gpu.fdid = 0;
    gpu.adapter = 0;

    if (tryNumber<int>(gpu.device, arg) && gpu.device < Gpu::deviceCount())
    {
        gpu.fdid = 0;
        gpu.adapter = 0;
        return;
    }

#ifdef __DIS_CLUSTER__
    if (tryDeviceId(arg, gpu.fdid, gpu.adapter))
    {
        gpu.device = -1;
        return;
    }
#endif

    throw error("`" + arg + "' is not a valid GPU device number");
}



static void qsInfo(std::ostringstream& s, const string& name, const string& argument, const string& info)
{
    using namespace std;
    s << "  " << left
        << setw(16) << name
        << setw(16) << argument
        << setw(40) << info
        << endl;
}



static void argInfo(std::ostringstream& s, const string& name, const string& argument, const string& info)
{
    qsInfo(s, ((name.length() == 1 ? "-" : "--") + name), argument, info);
}



static void argInfo(std::ostringstream& s, const string& name, const string& info)
{
    argInfo(s, name, "", info);
}



static string helpString(const string& progname)
{
    using namespace std;
    ostringstream s;

    s << "Usage: " << progname << " --ctrl <path> --queue <string>... --blocks=<count>" << endl;
#ifdef __DIS_CLUSTER__
    s << "   or: " << progname << " --ctrl <fdid>[:<ntb>] [--client] --queue <string>... --blocks=<count>" << endl;
#endif
    s << endl;

    s << "Arguments" << endl;
    argInfo(s, "help", "show this help");
    argInfo(s, "ctrl", "<path>", "path to NVM controller device");
#ifdef __DIS_CLUSTER__
    argInfo(s, "ctrl", "<fdid>[:<ntb>]", "NVM controller fabric ID and NTB adapter number");
    argInfo(s, "rpc", "request I/O queues from admin and do not reset controller");
    argInfo(s, "client", "alias for --rpc");
    argInfo(s, "segment", "<id>", "specify segment identifier offset (default 0)");
#endif
    argInfo(s, "queue", "<string>", "specify I/O queue (can be repeated), see queue specification");
    argInfo(s, "namespace", "<id>", "NVM namespace (default is 1)");
    argInfo(s, "ns", "<id>", "alias for --namespace");
    argInfo(s, "count", "<count>", "specify number of data units");
    argInfo(s, "offset", "<count>", "specify starting offset (default is 0)");
    argInfo(s, "blocks", "[<count>]", "specify data unit as blocks (default)");
    argInfo(s, "prps", "[<count>]", "specify data unit as PRPs/pages");
    argInfo(s, "pages", "[<count>]", "alias for --prps");
    argInfo(s, "bytes", "[<count>]", "specify data unit as bytes");
//    argInfo(s, "commands", "[<count>]", "specify data unit as number of I/O commands");
//    argInfo(s, "cmds", "[<count>]", "alias for --commands");
    argInfo(s, "random", "use random access instead of sequential access");
    argInfo(s, "parallel", "specify parallel access to separate buffers");
    argInfo(s, "shared", "same as parallel, but all queues access same buffer");
    argInfo(s, "gpu", "<device>", "host memory on local CUDA device instead of RAM");
#ifdef __DIS_CLUSTER__
    argInfo(s, "gpu", "<fdid>[:<ntb>]", "host memory on remote CUDA device instead of RAM (NB! must be borrowed)");
#endif
    argInfo(s, "bandwidth", "use a simpler algorithm for benchmarking bandwidth");
    argInfo(s, "bw", "alias for --bandwidth");
    argInfo(s, "stats", "[<file>]", "print benchmark statistics");
    argInfo(s, "info", "print information about transfers, buffers and queues");
    argInfo(s, "iterations", "<count>", "repeat benchmark <count> times (default is 1000)");
    argInfo(s, "outer", "<count>", "alias for --iterations");
    argInfo(s, "repeat", "<count>", "repeat commands <count> times (default is 1)");
    argInfo(s, "inner", "<count>", "alias for --repeat");
    argInfo(s, "outfile", "<filename>", "read from disk and write to file");
    argInfo(s, "output", "<filename>", "alias for --outfile");
    argInfo(s, "infile", "<filename>", "load file into buffer before reading from disk");
    argInfo(s, "input", "<filename>", "alias for --infile");
    argInfo(s, "write", "write to disk before reading (NB! destroys data on disk)");
    argInfo(s, "verify", "verify disk read/write");

    s << endl;
    s << "Queue specification string" << endl;
    s << "  The queue specification is a CSV string, where each parameter is formatted as key-value pair." << endl;
    s << "  For example: `no=1,cmds=32,prps=1,location=local' or `no=3,depth=1'" << endl;
    s << endl;
    s << "Possible queue specification key-value pairs:" << endl;
    qsInfo(s, "no", "<number>", "unique I/O queue number (required)");
    qsInfo(s, "id", "<number>", "alias for no");
    qsInfo(s, "cmds", "<commands>", "number of unsubmitted I/O commands (defaults to controller's maximum)");
    qsInfo(s, "depth", "<commands>", "alias for cmds");
    qsInfo(s, "prps", "<pages>", "data transfer size in number of PRPs/pages (defaults to controller's maximum for sequential and 1 for random)");
    qsInfo(s, "pages", "<pages>", "alias for prps");
    qsInfo(s, "location", "host", "host submission queue in local RAM (default)");
    qsInfo(s, "location", "local", "alias for specifying host location");
#ifdef __DIS_CLUSTER__
    qsInfo(s, "location", "remote", "host submission queue in remote RAM close to controller");
    qsInfo(s, "location", "<fdid>[:<ntb>]", "host submission queue in remote CUDA device memory (NB! must be borrwed)");
#endif
    s << endl;

    return s.str();
}



Settings::Settings()
{
    ctrl.fdid = 0;
    ctrl.adapter = 0;
    gpu.device = -1;
    gpu.fdid = 0;
    gpu.adapter = 0;
    latency = true;
    nvmNamespace = 1;
    outerIterations = 1000;
    innerIterations = 1;
    count = 0;
    offset = 0;
    unit = AccessUnit::BLOCKS;
    parallel = false;
    shared = false;
    sequential = true;
    random = false;
    stats = false;
    manager = true;
    segmentOffset = 0;
    write = false;
    verify = false;
    transferInfo = false;
}



static const option options[] = {
    { .name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h' },
    { .name = "disk", .has_arg = required_argument, .flag = nullptr, .val = 'd' },
    { .name = "dev", .has_arg = required_argument, .flag = nullptr, .val = 'd' },
    { .name = "ctrl", .has_arg = required_argument, .flag = nullptr, .val = 'd' },
    { .name = "rpc-client", .has_arg = no_argument, .flag = nullptr, .val = 'm' },
    { .name = "client", .has_arg = no_argument, .flag = nullptr, .val = 'm' },
    { .name = "rpc", .has_arg = no_argument, .flag = nullptr, .val = 'm' },
    { .name = "gpu", .has_arg = required_argument, .flag = nullptr, .val = 'g' },
    { .name = "queue", .has_arg = required_argument, .flag = nullptr, .val = 'q' },
    { .name = "seg", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "segment", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "segment-id", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "segment-offset", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "segoff", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "soff", .has_arg = required_argument, .flag = nullptr, .val = 's' },
    { .name = "size", .has_arg = required_argument, .flag = nullptr, .val = 'c' },
    { .name = "count", .has_arg = required_argument, .flag = nullptr, .val = 'c' },
    { .name = "offset", .has_arg = required_argument, .flag = nullptr, .val = 'o' },
    { .name = "offs", .has_arg = required_argument, .flag = nullptr, .val = 'o' },
    { .name = "off", .has_arg = required_argument, .flag = nullptr, .val = 'o' },
    { .name = "namespace", .has_arg = required_argument, .flag = nullptr, .val = 'n' },
    { .name = "ns", .has_arg = required_argument, .flag = nullptr, .val = 'n' },
    { .name = "blocks", .has_arg = optional_argument, .flag = nullptr, .val = 'b' },
    { .name = "blks", .has_arg = optional_argument, .flag = nullptr, .val = 'b' },
    { .name = "pages", .has_arg = optional_argument, .flag = nullptr, .val = 3 },
    { .name = "prps", .has_arg = optional_argument, .flag = nullptr, .val = 3 },
    { .name = "bytes", .has_arg = optional_argument, .flag = nullptr, .val = 1 },
    { .name = "commands", .has_arg = optional_argument, .flag = nullptr, .val = 'C' },
    { .name = "cmds", .has_arg = optional_argument, .flag = nullptr, .val = 'C' },
    { .name = "statistics", .has_arg = optional_argument, .flag = nullptr, .val = 'S' },
    { .name = "stats", .has_arg = optional_argument, .flag = nullptr, .val = 'S' },
    { .name = "bandwidth", .has_arg = no_argument, .flag = nullptr, .val = 'B' },
    { .name = "bw", .has_arg = no_argument, .flag = nullptr, .val = 'B' },
    { .name = "random", .has_arg = no_argument, .flag = nullptr, .val = 'r' },
    { .name = "parallel", .has_arg = no_argument, .flag = nullptr, .val = 'p' },
    { .name = "shared", .has_arg = no_argument, .flag = nullptr, .val = 'P' },
    { .name = "iterations", .has_arg = required_argument, .flag = nullptr, .val = 'i' },
    { .name = "outer", .has_arg = required_argument, .flag = nullptr, .val = 'i' },
    { .name = "loops", .has_arg = required_argument, .flag = nullptr, .val = 'i' },
    { .name = "repeat", .has_arg = required_argument, .flag = nullptr, .val = 'I' },
    { .name = "inner", .has_arg = required_argument, .flag = nullptr, .val = 'I' },
    { .name = "outfile", .has_arg = required_argument, .flag = nullptr, .val = 'R' },
    { .name = "output", .has_arg = required_argument, .flag = nullptr, .val = 'R' },
    { .name = "of", .has_arg = required_argument, .flag = nullptr, .val = 'R' },
    { .name = "infile", .has_arg = required_argument, .flag = nullptr, .val = 'W' },
    { .name = "input", .has_arg = required_argument, .flag = nullptr, .val = 'W' },
    { .name = "if", .has_arg = required_argument, .flag = nullptr, .val = 'W' },
    { .name = "write", .has_arg = no_argument, .flag = nullptr, .val = 'w' },
    { .name = "verify", .has_arg = no_argument, .flag = nullptr, .val = 'v' },
    { .name = "info", .has_arg = no_argument, .flag = nullptr, .val = 't' },
    { .name = nullptr, .has_arg = no_argument, .flag = nullptr, .val = 0 }
};


void Settings::parseArguments(int argc, char** argv)
{
    int index;
    int option;

    if (argc == 1)
    {
        throw error(helpString(argv[0]));
    }

    // Build option string for getopt
    string optstr{':'};
    for (size_t i = 0; options[i].name != nullptr; ++i)
    {
        optstr += options[i].val;
        if (options[i].has_arg != no_argument)
        {
            optstr += ":";
        }
    }

    try
    {
        // Do getopt magic
        while ((option = getopt_long(argc, argv, optstr.c_str(), options, &index)) != -1)
        {
            switch (option)
            {
                case 'h':
                    throw helpString(argv[0]);

                case '?':
                    throw error("Option `" + string(argv[optind - 1]) + "' is unknown");

                case ':':
                    throw error("Missing argument value for option `" + string(argv[optind - 1]) + "'");

                case 'd':
                    setController(ctrl, optarg);
                    break;

                case 'g':
                    setGpu(gpu, optarg);
                    break;

                case 'q':
                    if (optarg)
                    {
                        queues.emplace_back(optarg);
                    }
                    else
                    {
                        queues.emplace_back();
                    }
                    break;

                case 'm':
                    manager = false;
                    break;

                case 't':
                    transferInfo = true;
                    break;

                case 'B':
                    latency = false;
                    break;

                case 'S':
                    stats = true;
                    statsFilename.clear();
                    if (optarg)
                    {
                        statsFilename = optarg;
                    }
                    break;

                case 'W':
                    inputFilename = optarg;
                    break;

                case 'R':
                    outputFilename = optarg;
                    break;

                case 1:
                    unit = AccessUnit::BYTES;
                    if (optarg != nullptr)
                    {
                        count = parseNumber(optarg);
                    }
                    break;

                case 'C':
                    unit = AccessUnit::COMMANDS;
                    if (optarg)
                    {
                        count = parseNumber(optarg);
                    }
                    break;

                case 'b':
                    unit = AccessUnit::BLOCKS;
                    if (optarg != nullptr)
                    {
                        count = parseNumber(optarg);
                    }
                    break;

                case 3:
                    unit = AccessUnit::PAGES;
                    if (optarg)
                    {
                        count = parseNumber(optarg);
                    }
                    break;

                case 'c':
                    count = parseNumber(optarg);
                    break;

                case 's':
                    segmentOffset = parseNumber(optarg);
                    break;

                case 'o':
                    offset = parseNumber(optarg);
                    break;

                case 'i':
                    outerIterations = parseNumber(optarg);
                    if (outerIterations == 0)
                    {
                        throw error("Number of iterations must be at least one");
                    }
                    break;

                case 'I':
                    innerIterations = parseNumber(optarg);
                    if (innerIterations == 0)
                    {
                        throw error("Number of repeats must be at least one");
                    }
                    break;

                case 'n':
                    nvmNamespace = parseNumber(optarg);
                    break;

                case 'p':
                    parallel = true;
                    break;

                case 'P':
                    parallel = true;
                    shared = true;
                    break;

                case 'r':
                    random = true;
                    break;

                case 'w':
                    write = true;
                    break;

                case 'v':
                    verify = true;
                    break;
            }
        }
    }
    catch (const error& e)
    {
        throw error("Argument error: " + string(e.what()));
    }

    if (ctrl.path.empty() && ctrl.fdid == 0)
    {
        throw error("No controller specified");
    }

    if (count == 0)
    {
        throw error("Count is not specified");
    }

    if (queues.empty())
    {
        throw error("No I/O queues are specified");
    }

    if (verify && inputFilename.empty())
    {
        throw error("Input file must be specified in order to verify");
    }
}

