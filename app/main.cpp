#include "snapsentinel/benchmark/data_path_experiment.hpp"

#include <charconv>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::size_t parsePositiveSize(const std::string_view option, const std::string_view value) {
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return parsed;
}

std::size_t parseNonNegativeSize(
    const std::string_view option,
    const std::string_view value
) {
    std::size_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
    }
    return parsed;
}

snapsentinel::benchmark::DataPathExperimentOptions parseOptions(
    const int argc,
    char* argv[]
) {
    snapsentinel::benchmark::DataPathExperimentOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--frames" || argument == "--iterations" ||
            argument == "--width" || argument == "--height" ||
            argument == "--warmup-frames") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            const std::string_view text(argv[++index]);
            const std::size_t value = argument == "--warmup-frames"
                ? parseNonNegativeSize(argument, text)
                : parsePositiveSize(argument, text);
            if (argument == "--frames") {
                options.frameCount = value;
            } else if (argument == "--iterations") {
                options.iterations = value;
            } else if (argument == "--width") {
                options.width = value;
            } else if (argument == "--height") {
                options.height = value;
            } else {
                options.warmupFrames = value;
            }
        } else if (argument == "--output-csv") {
            if (index + 1 >= argc || argv[index + 1][0] == '\0') {
                throw std::invalid_argument("--output-csv requires a path");
            }
            options.csvPath = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            throw std::invalid_argument(
                "Usage: snapsentinel_runner [--frames <count>] [--iterations <count>] "
                "[--width <pixels>] [--height <pixels>] [--warmup-frames <count>] "
                "[--output-csv <path>]"
            );
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(argument));
        }
    }
    return options;
}

}  // namespace

int main(const int argc, char* argv[]) {
    try {
        const snapsentinel::benchmark::DataPathExperimentOptions options =
            parseOptions(argc, argv);
        static_cast<void>(snapsentinel::benchmark::runDataPathExperiment(options));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SnapSentinel runner failed: " << error.what() << '\n';
        return 1;
    }
}
