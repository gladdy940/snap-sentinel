#include "snapsentinel/pipeline/benchmark_runner.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace snapsentinel::pipeline {
namespace {

using BenchmarkClock = std::chrono::high_resolution_clock;
constexpr auto kSampleInterval = std::chrono::milliseconds(100);
constexpr auto kMaximumDuration = std::chrono::minutes(30);

struct CpuCounters final {
    std::uint64_t active = 0;
    std::uint64_t total = 0;
};

struct PlatformSample final {
    double cpuUtilizationPct = 0.0;
    std::optional<double> temperatureCelsius;
    bool frequencyThrottled = false;
};

struct LatencySample final {
    std::uint64_t frameId;
    double dispatchLatencyMs;
    double completionLatencyMs;
};

struct InFlightMeasurement final {
    std::uint64_t frameId;
    BenchmarkClock::time_point startTime;
    double dispatchLatencyMs = 0.0;
    double completionLatencyMs = 0.0;
    bool dispatchReady = false;
    bool completionReceived = false;
    bool cancelled = false;
};

std::optional<CpuCounters> readCpuCounters() {
#if defined(__linux__)
    std::ifstream statFile("/proc/stat");
    std::string cpuLabel;
    CpuCounters counters;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;
    if (!(statFile >> cpuLabel >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) ||
        cpuLabel != "cpu") {
        return std::nullopt;
    }
    counters.active = user + nice + system + irq + softirq + steal;
    counters.total = counters.active + idle + iowait;
    return counters;
#else
    return std::nullopt;
#endif
}

std::optional<double> readThermalTemperature() {
#if defined(__linux__)
    constexpr char thermalRoot[] = "/sys/class/thermal";
    std::error_code error;
    const std::filesystem::directory_iterator end;
    double hottestCelsius = -std::numeric_limits<double>::infinity();
    for (std::filesystem::directory_iterator iterator(
             thermalRoot,
             std::filesystem::directory_options::skip_permission_denied,
             error);
         !error && iterator != end;
         iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        if (name.rfind("thermal_zone", 0) != 0) {
            continue;
        }

        std::ifstream temperatureFile(iterator->path() / "temp");
        long long millidegrees = 0;
        if (temperatureFile >> millidegrees) {
            hottestCelsius = std::max(hottestCelsius, static_cast<double>(millidegrees) / 1000.0);
        }
    }
    if (std::isfinite(hottestCelsius)) {
        return hottestCelsius;
    }
#endif
    return std::nullopt;
}

#if defined(__linux__)
std::optional<std::uint64_t> readFrequency(const std::filesystem::path& path) {
    std::ifstream frequencyFile(path);
    std::uint64_t value = 0;
    if (frequencyFile >> value) {
        return value;
    }
    return std::nullopt;
}

bool readFrequencyThrottling() {
    constexpr char coolingRoot[] = "/sys/class/thermal";
    {
        std::error_code error;
        const std::filesystem::directory_iterator end;
        for (std::filesystem::directory_iterator iterator(
                 coolingRoot,
                 std::filesystem::directory_options::skip_permission_denied,
                 error);
             !error && iterator != end;
             iterator.increment(error)) {
            const std::string name = iterator->path().filename().string();
            if (name.rfind("cooling_device", 0) != 0) {
                continue;
            }

            std::ifstream typeFile(iterator->path() / "type");
            std::string type;
            std::ifstream stateFile(iterator->path() / "cur_state");
            unsigned int state = 0;
            if ((typeFile >> type) && (stateFile >> state) && state > 0 &&
                (type.find("cpu") != std::string::npos ||
                 type.find("CPU") != std::string::npos ||
                 type.find("Processor") != std::string::npos)) {
                return true;
            }
        }
    }

    constexpr char policyRoot[] = "/sys/devices/system/cpu/cpufreq";
    std::error_code error;
    const std::filesystem::directory_iterator end;
    for (std::filesystem::directory_iterator iterator(
             policyRoot,
             std::filesystem::directory_options::skip_permission_denied,
             error);
         !error && iterator != end;
         iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        if (name.rfind("policy", 0) != 0) {
            continue;
        }
        const auto policyMaximum =
            readFrequency(iterator->path() / "scaling_max_freq");
        const auto hardwareMaximum =
            readFrequency(iterator->path() / "cpuinfo_max_freq");
        if (policyMaximum.has_value() && hardwareMaximum.has_value() &&
            *policyMaximum < *hardwareMaximum) {
            return true;
        }
    }
    return false;
}
#else
bool readFrequencyThrottling() {
    return false;
}
#endif

PlatformSample readPlatformSample(const std::optional<CpuCounters>& previousCounters) {
    PlatformSample result;
    result.temperatureCelsius = readThermalTemperature();
    result.frequencyThrottled = readFrequencyThrottling();

    const std::optional<CpuCounters> currentCounters = readCpuCounters();
    if (previousCounters.has_value() && currentCounters.has_value()) {
        if (currentCounters->total < previousCounters->total ||
            currentCounters->active < previousCounters->active) {
            return result;
        }
        const std::uint64_t totalDelta = currentCounters->total - previousCounters->total;
        const std::uint64_t activeDelta = currentCounters->active - previousCounters->active;
        if (totalDelta > 0) {
            result.cpuUtilizationPct =
                100.0 * static_cast<double>(activeDelta) / static_cast<double>(totalDelta);
        }
    }
    return result;
}

std::int64_t timestampMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

double percentile99(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t index =
        static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(sorted.size()))) - 1;
    return sorted[index];
}

}  // namespace

BenchmarkOptions BenchmarkOptions::parseCommandLine(
    const int argc,
    const char* const argv[]
) {
    BenchmarkOptions options;
    bool durationProvided = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--duration-sec") {
            if (durationProvided || index + 1 >= argc) {
                throw std::invalid_argument("--duration-sec requires exactly one positive value");
            }
            ++index;
            std::size_t parsedLength = 0;
            long long seconds = 0;
            try {
                seconds = std::stoll(argv[index], &parsedLength);
            } catch (const std::exception&) {
                throw std::invalid_argument("--duration-sec must be an integer from 1 to 1800");
            }
            if (parsedLength != std::string(argv[index]).size() ||
                seconds < 1 ||
                seconds > std::chrono::duration_cast<std::chrono::seconds>(
                              kMaximumDuration
                          ).count()) {
                throw std::invalid_argument("--duration-sec must be an integer from 1 to 1800");
            }
            options.duration = std::chrono::seconds(seconds);
            durationProvided = true;
        } else if (argument == "--output-csv") {
            if (index + 1 >= argc || argv[index + 1][0] == '\0') {
                throw std::invalid_argument("--output-csv requires a file path");
            }
            options.csvPath = argv[++index];
        } else if (argument == "--help" || argument == "-h") {
            throw std::invalid_argument(usage(argc > 0 ? argv[0] : "snapsentinel_runner"));
        } else {
            throw std::invalid_argument("Unknown benchmark option: " + std::string(argument));
        }
    }
    return options;
}

std::string BenchmarkOptions::usage(const char* const executableName) {
    std::ostringstream message;
    message << "Usage: " << executableName
            << " --duration-sec <1..1800> [--output-csv <path>]";
    return message.str();
}

BenchmarkRunner::BenchmarkRunner(InferenceSubmitter submitter)
    : submitter_(std::move(submitter)) {
    if (!submitter_) {
        throw std::invalid_argument("Benchmark inference submitter must not be empty");
    }
}

BenchmarkSummary BenchmarkRunner::runCommandLine(
    const int argc,
    const char* const argv[],
    InferenceSubmitter submitter
) {
    const BenchmarkOptions options = BenchmarkOptions::parseCommandLine(argc, argv);
    return BenchmarkRunner(std::move(submitter)).run(options);
}

BenchmarkSummary BenchmarkRunner::run(const BenchmarkOptions& options) const {
    if (options.duration <= std::chrono::seconds::zero() ||
        options.duration > kMaximumDuration) {
        throw std::invalid_argument("Benchmark duration must be between 1 second and 30 minutes");
    }
    if (options.csvPath.empty()) {
        throw std::invalid_argument("Benchmark CSV path must not be empty");
    }

    std::ofstream csv(options.csvPath, std::ios::out | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("Unable to open benchmark CSV: " + options.csvPath.string());
    }
    csv << "Timestamp_ms,Frame_ID,Latency_ms,Dispatch_Latency_ms,CPU_Util_Pct,"
           "Thermal_Zone_C,CPU_Throttled\n";
    csv << std::fixed << std::setprecision(3);

    std::mutex completionMutex;
    std::condition_variable completionCondition;
    std::vector<LatencySample> latencySamples;
    std::size_t pending = 0;
    std::uint64_t frameId = 0;
    std::uint64_t latestCompletedFrameId = 0;
    double latestCompletionLatencyMs = 0.0;
    double latestDispatchLatencyMs = 0.0;
    std::exception_ptr callbackFailure;
    const auto finishMeasurement = [&](const std::shared_ptr<InFlightMeasurement>& measurement) {
        if (measurement->dispatchReady && measurement->completionReceived) {
            latencySamples.push_back(LatencySample{
                measurement->frameId,
                measurement->dispatchLatencyMs,
                measurement->completionLatencyMs
            });
            latestCompletedFrameId = measurement->frameId;
            latestDispatchLatencyMs = measurement->dispatchLatencyMs;
            latestCompletionLatencyMs = measurement->completionLatencyMs;
        }
    };

    const auto start = BenchmarkClock::now();
    const auto deadline = start + options.duration;
    auto nextSample = start;
    std::optional<CpuCounters> previousCpuCounters = readCpuCounters();
    double minimumTemperature = std::numeric_limits<double>::infinity();
    double maximumTemperature = -std::numeric_limits<double>::infinity();

    try {
        while (BenchmarkClock::now() < deadline) {
            const std::uint64_t currentFrameId = frameId++;
            const auto dispatchStart = BenchmarkClock::now();
            auto measurement = std::make_shared<InFlightMeasurement>(
                InFlightMeasurement{currentFrameId, dispatchStart}
            );
            {
                std::lock_guard lock(completionMutex);
                ++pending;
            }

            try {
                submitter_(currentFrameId, [&, measurement] {
                    const double completionLatencyMs =
                        std::chrono::duration<double, std::milli>(
                            BenchmarkClock::now() - measurement->startTime
                        ).count();
                    std::lock_guard lock(completionMutex);
                    if (measurement->cancelled || measurement->completionReceived) {
                        return;
                    }
                    measurement->completionReceived = true;
                    measurement->completionLatencyMs = completionLatencyMs;
                    try {
                        finishMeasurement(measurement);
                    } catch (...) {
                        callbackFailure = std::current_exception();
                    }
                    --pending;
                    completionCondition.notify_all();
                });
            } catch (...) {
                std::lock_guard lock(completionMutex);
                if (!measurement->completionReceived) {
                    measurement->cancelled = true;
                    --pending;
                    completionCondition.notify_all();
                }
                throw;
            }

            const double dispatchLatencyMs =
                std::chrono::duration<double, std::milli>(
                    BenchmarkClock::now() - dispatchStart
                ).count();
            {
                std::lock_guard lock(completionMutex);
                measurement->dispatchReady = true;
                measurement->dispatchLatencyMs = dispatchLatencyMs;
                finishMeasurement(measurement);
            }

            const auto now = BenchmarkClock::now();
            if (now >= nextSample) {
                const PlatformSample platformSample = readPlatformSample(previousCpuCounters);
                previousCpuCounters = readCpuCounters();
                if (platformSample.temperatureCelsius.has_value()) {
                    minimumTemperature =
                        std::min(minimumTemperature, *platformSample.temperatureCelsius);
                    maximumTemperature =
                        std::max(maximumTemperature, *platformSample.temperatureCelsius);
                }

                double completionLatency = 0.0;
                double dispatchLatency = 0.0;
                std::uint64_t sampledFrameId = 0;
                {
                    std::lock_guard lock(completionMutex);
                    completionLatency = latestCompletionLatencyMs;
                    dispatchLatency = latestDispatchLatencyMs;
                    sampledFrameId = latestCompletedFrameId;
                }
                csv << timestampMilliseconds() << ',' << sampledFrameId << ','
                    << completionLatency << ',' << dispatchLatency << ','
                    << platformSample.cpuUtilizationPct << ',';
                if (platformSample.temperatureCelsius.has_value()) {
                    csv << *platformSample.temperatureCelsius;
                }
                csv << ',' << (platformSample.frequencyThrottled ? 1 : 0) << '\n';
                csv.flush();
                if (!csv) {
                    throw std::runtime_error("Failed while writing benchmark CSV");
                }
                nextSample += kSampleInterval;
                while (nextSample <= now) {
                    nextSample += kSampleInterval;
                }
            }
        }
    } catch (...) {
        std::unique_lock lock(completionMutex);
        completionCondition.wait(lock, [&pending] { return pending == 0; });
        throw;
    }

    {
        std::unique_lock lock(completionMutex);
        completionCondition.wait(lock, [&pending] { return pending == 0; });
    }
    if (callbackFailure != nullptr) {
        std::rethrow_exception(callbackFailure);
    }
    csv.flush();
    if (!csv) {
        throw std::runtime_error("Failed to flush benchmark CSV");
    }

    std::vector<double> completionLatencies;
    completionLatencies.reserve(latencySamples.size());
    double totalLatency = 0.0;
    for (const LatencySample& sample : latencySamples) {
        totalLatency += sample.completionLatencyMs;
        completionLatencies.push_back(sample.completionLatencyMs);
    }

    BenchmarkSummary summary;
    summary.completedFrames = static_cast<std::uint64_t>(latencySamples.size());
    if (!latencySamples.empty()) {
        summary.meanLatencyMs = totalLatency / static_cast<double>(latencySamples.size());
        summary.p99LatencyMs = percentile99(completionLatencies);
    }
    if (std::isfinite(minimumTemperature) && std::isfinite(maximumTemperature)) {
        summary.maxThermalShiftC = maximumTemperature - minimumTemperature;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "Benchmark completed frames: " << summary.completedFrames << '\n'
              << "Mean Latency: " << summary.meanLatencyMs << " ms\n"
              << "P99 Latency: " << summary.p99LatencyMs << " ms\n"
              << "Max Thermal Shift: " << summary.maxThermalShiftC << " C\n";
    return summary;
}

}  // namespace snapsentinel::pipeline
