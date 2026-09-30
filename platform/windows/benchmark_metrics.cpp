#include "snapsentinel/benchmark/platform_metrics.hpp"

namespace snapsentinel::benchmark {
namespace {

class WindowsBenchmarkMetrics final : public PlatformMetricsProvider {
public:
    [[nodiscard]] PlatformMetrics sample() override {
        return {};
    }
};

}  // namespace

std::unique_ptr<PlatformMetricsProvider> createPlatformMetricsProvider() {
    return std::make_unique<WindowsBenchmarkMetrics>();
}

}  // namespace snapsentinel::benchmark
