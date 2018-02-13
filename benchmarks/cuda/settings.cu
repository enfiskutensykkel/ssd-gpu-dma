#include <cuda.h>
#include "settings.h"
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <functional>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <getopt.h>

struct OptionIface;
using std::string;
using std::vector;
using std::make_shared;
typedef std::shared_ptr<OptionIface>  OptionPtr;
typedef std::map<int, OptionPtr> OptionMap;


struct OptionIface
{
    string          type;
    string          name;
    string          description;
    string          defaultValue;
    int             hasArgument;

    virtual ~OptionIface() = default;

    OptionIface(const string& type, const string& name, const string& description)
        : type(type), name(name), description(description), hasArgument(no_argument) { }

    OptionIface(const string& type, const string& name, const string& description, const string& dvalue)
        : type(type), name(name), description(description), defaultValue(dvalue), hasArgument(no_argument) { }

    virtual void parseArgument(const char* optstr, const char* optarg) = 0;

    virtual void throwError(const char*, const char* optarg) const
    {
        throw string("Option ") + name + string(" expects a ") + type + string(", but got `") + optarg + string("'");
    }
};


template <typename T>
struct Option: public OptionIface
{
    T&              value;

    Option(T& value, const string& type, const string& name, const string& description)
        : OptionIface(type, name, description)
        , value(value)
    {
        hasArgument = required_argument;
    }

    Option(T& value, const string& type, const string& name, const string& description, const string& dvalue)
        : OptionIface(type, name, description, dvalue)
        , value(value)
    {
        hasArgument = required_argument;
    }

    void parseArgument(const char* optstr, const char* optarg) override;
};


template <>
void Option<uint32_t>::parseArgument(const char* optstr, const char* optarg)
{
    char* endptr = nullptr;

    value = std::strtoul(optarg, &endptr, 0);

    if (endptr == nullptr || *endptr != '\0')
    {
        throwError(optstr, optarg);
    }
}


template <>
void Option<uint64_t>::parseArgument(const char* optstr, const char* optarg)
{
    char* endptr = nullptr;

    value = std::strtoul(optarg, &endptr, 0);

    if (endptr == nullptr || *endptr != '\0')
    {
        throwError(optstr, optarg);
    }
}


template <>
void Option<bool>::parseArgument(const char* optstr, const char* optarg)
{
    string str(optarg);
    std::transform(str.begin(), str.end(), str.begin(), std::ptr_fun<int, int>(std::tolower));

    if (str == "false" || str == "0")
    {
        value = false;
    }
    else if (str == "true" || str == "1")
    {
        value = true;
    }
    else
    {
        throwError(optstr, optarg);
    }
}


template <>
void Option<const char*>::parseArgument(const char*, const char* optarg)
{
    value = optarg;
}


struct Flag: public Option<bool>
{
    Flag(bool& value, const string& name, const string& description)
        : Option<bool>(value, "bool", name, description, "false")
    {
        hasArgument = optional_argument;
    }

    void throwError(const char* str, const char* arg) const override
    {
        throw string("Option ") + name + string(" must be set to true or false");
    }

    void parseArgument(const char* str, const char* arg) override
    {
        if (arg == nullptr)
        {
            value = true;
        }

        Option<bool>::parseArgument(str, arg);
    }
};


struct Range: public Option<uint64_t>
{
    uint64_t      lower;
    uint64_t      upper;

    Range(uint64_t& value, uint64_t lo, uint64_t hi, const string& name, const string& description, const string& dv)
        : Option<uint64_t>(value, "count", name, description, dv)
        , lower(lo)
        , upper(hi)
    { }

    void throwError(const char* str, const char* arg) const override
    {
        if (upper != 0 && lower != 0)
        {
            throw string("Option ") + name + string(" expects a value between ") + std::to_string(lower) + " and " + std::to_string(upper);
        }
        else if (lower != 0)
        {
            throw string("Option ") + name + string(" must be at least ") + std::to_string(lower);
        }
        throw string("Option ") + name + string(" must lower than ") + std::to_string(upper);
    }

    void parseArgument(const char* optstr, const char* optarg) override
    {
        Option<uint64_t>::parseArgument(optstr, optarg);

        if (lower != 0 && value < lower)
        {
            throwError(optstr, optarg);
        }

        if (upper != 0 && value > upper)
        {
            throwError(optstr, optarg);
        }
    }
};



static string usageString(const string& name)
{
    return "Usage: " + name + " --ctrl=identifier\n"
        +  "   or: " + name + " --block-device=path";
}



static string helpString(const string& name, OptionMap& options)
{
    using namespace std;
    ostringstream s;

    s << usageString(name) << endl;
    s << endl;

    s << "" << left
        << setw(16) << "OPTION"
        << setw(2) << " "
        << setw(16) << "TYPE" 
        << setw(10) << "DEFAULT"
        << setw(36) << "DESCRIPTION" 
        << endl;
    for (const auto& optPair: options)
    {
        const auto& opt = optPair.second;
        s << "  " << left
            << setw(16) << ((opt->name.length() == 1 ? "-" : "--") + opt->name)
            << setw(16) << opt->type
            << setw(10) << opt->defaultValue
            << setw(36) << opt->description
            << endl;
    }

    return s.str();
}


static void createLongOptions(vector<option>& options, string& optionString, const OptionMap& parsers)
{
    options.push_back(option{ .name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h' });
    optionString = ":h";

    for (const auto& parserPair: parsers)
    {
        int shortOpt = parserPair.first;
        const OptionPtr& parser = parserPair.second;

        option opt;
        opt.name = parser->name.c_str();
        opt.has_arg = parser->hasArgument;
        opt.flag = nullptr;
        opt.val = shortOpt;

        options.push_back(opt);

        if ('0' <= shortOpt && shortOpt <= 'z')
        {
            optionString += (char) shortOpt;
            if (parser->hasArgument == required_argument)
            {
                optionString += ":";
            }
            else if (parser->hasArgument == optional_argument)
            {
                optionString += "::";
            }
        }
    }

    options.push_back(option{ .name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0 });
}


static void verifyCudaDevice(int device)
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        throw string("Unexpected error: ") + cudaGetErrorString(err);
    }

    if (device < 0 || device >= deviceCount)
    {
        throw string("Invalid CUDA device: ") + std::to_string(device);
    }
}


static void verifyNumberOfThreads(size_t numThreads)
{
    size_t i = 0;

    while ((1ULL << i) <= 32)
    {
        if ((1ULL << i) == numThreads)
        {
            return;
        }

        ++i;
    }

    throw string("Invalid number of threads, must be a power of 2");
}


void Settings::parseArguments(int argc, char** argv)
{
    OptionMap parsers = {
#ifdef __DIS_CLUSTER__
        {'c', OptionPtr(new Option<uint64_t>(controllerId, "identifier", "ctrl", "NVM controller device identifier"))},
        {'a', OptionPtr(new Option<uint32_t>(adapter, "number", "adapter", "DIS adapter number", "0"))},
        {'S', OptionPtr(new Option<uint32_t>(segmentId, "offset", "segment", "DIS segment identifier offset", "0"))},
#else
        {'c', OptionPtr(new Option<const char*>(controllerPath, "path", "ctrl", "NVM controller device path"))},
#endif
        {'g', OptionPtr(new Option<uint32_t>(cudaDevice, "number", "gpu", "specify CUDA device", "0"))},
        {'i', OptionPtr(new Option<uint32_t>(nvmNamespace, "identifier", "namespace", "NVM namespace identifier", "0"))},
        {'B', OptionPtr(new Flag(doubleBuffered, "double-buffer", "double buffer disk reads"))},
        {'n', OptionPtr(new Range(numChunks, 1, 0, "chunks", "number of chunks per thread", "32"))},
        {'p', OptionPtr(new Range(numPages, 1, 0, "pages", "number of pages per chunk", "1"))},
        {'t', OptionPtr(new Range(numThreads, 1, 32, "threads", "number of CUDA threads", "32"))},
        {'o', OptionPtr(new Option<const char*>(output, "path", "output", "output read data to file"))},
        {'s', OptionPtr(new Option<uint64_t>(startBlock, "offset", "offset", "number of blocks to offset", "0"))},
        {'b', OptionPtr(new Option<const char*>(blockDevicePath, "path", "block-device", "path to block device"))}
    };

    string optionString;
    vector<option> options;
    createLongOptions(options, optionString, parsers);

    int index;
    int option;
    OptionMap::iterator parser;

    while ((option = getopt_long(argc, argv, optionString.c_str(), &options[0], &index)) != -1)
    {
        switch (option)
        {
            case '?':
                throw string("Unknown option: `") + argv[optind - 1] + string("'");

            case ':':
                throw string("Missing argument for option ") + argv[optind - 1];

            case 'h':
                throw helpString(argv[0], parsers);

            default:
                parser = parsers.find(option);
                if (parser == parsers.end())
                {
                    throw string("Unknown option: `") + argv[optind - 1] + string("'");
                }
                parser->second->parseArgument(argv[optind - 1], optarg);
                break;
        }
    }

#ifdef __DIS_CLUSTER__
    if (blockDevicePath == nullptr && controllerId == 0)
    {
        throw string("No block device or NVM controller specified");
    }
#else
    if (blockDevicePath == nullptr && controllerPath == nullptr)
    {
        throw string("No block device or NVM controller specified");
    }
#endif

    verifyCudaDevice(cudaDevice);
    verifyNumberOfThreads(numThreads);
}


Settings::Settings()
{
    cudaDevice = 0;
    blockDevicePath = nullptr;
    controllerPath = nullptr;
    controllerId = 0;
    adapter = 0;
    segmentId = 0;
    nvmNamespace = 1;
    doubleBuffered = false;
    numChunks = 32;
    numPages = 1;
    startBlock = 0;
    stats = false;
    output = nullptr;
    numThreads = 32;
}


