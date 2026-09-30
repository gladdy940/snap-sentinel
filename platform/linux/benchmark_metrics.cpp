#include "snapsentinel/benchmark/platform_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

namespace snapsentinel::benchmark {
namespace {

struct CpuCounters final {
    std::uint64_t active = 0;
    std::uint64_t total = 0;
};

std::optional<CpuCounters> readCpuCounters() {
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
}

std::optional<double> readThermalTemperature() {
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
            hottestCelsius =
                std::max(hottestCelsius, static_cast<double>(millidegrees) / 1000.0);
        }
    }
    if (std::isfinite(hottestCelsius)) {
        return hottestCelsius;
    }
    return std::nullopt;
}

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

class LinuxBenchmarkMetrics final : public PlatformMetricsProvider {
public:
    [[nodiscard]] PlatformMetrics sample() override {
        PlatformMetrics result;
        result.temperatureCelsius = readThermalTemperature();
        result.cpuThrottled = readFrequencyThrottling();

        const std::optional<CpuCounters> current = readCpuCounters();
        if (previousCounters_.has_value() && current.has_value() &&
            current->total >= previousCounters_->total &&
            current->active >= previousCounters_->active) {
            const std::uint64_t totalDelta = current->total - previousCounters_->total;
            const std::uint64_t activeDelta = current->active - previousCounters_->active;
            if (totalDelta > 0) {
                result.cpuUtilizationPct =
                    100.0 * static_cast<double>(activeDelta) / static_cast<double>(totalDelta);
            }
        }
        previousCounters_ = current;
        return result;
    }

private:
    std::optional<CpuCounters> previousCounters_ = readCpuCounters();
};

}  // namespace

std::unique_ptr<PlatformMetricsProvider> createPlatformMetricsProvider() {
    return std::make_unique<LinuxBenchmarkMetrics>();
}

}  // namespace snapsentinel::benchmark
