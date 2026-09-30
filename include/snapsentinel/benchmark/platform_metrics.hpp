#pragma once

#include <memory>
#include <optional>

namespace snapsentinel::benchmark {

struct PlatformMetrics final {
    std::optional<double> cpuUtilizationPct;
    std::optional<double> temperatureCelsius;
    std::optional<bool> cpuThrottled;
};

class PlatformMetricsProvider {
public:
    virtual ~PlatformMetricsProvider() = default;

    [[nodiscard]] virtual PlatformMetrics sample() = 0;
};

[[nodiscard]] std::unique_ptr<PlatformMetricsProvider> createPlatformMetricsProvider();

}  // namespace snapsentinel::benchmark
