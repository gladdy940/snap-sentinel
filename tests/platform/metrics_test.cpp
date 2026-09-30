#include "snapsentinel/benchmark/platform_metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

namespace snapsentinel::benchmark {
namespace {

TEST(PlatformMetricsTest, ProviderReportsOnlyAvailableMeasurements) {
    std::unique_ptr<PlatformMetricsProvider> provider = createPlatformMetricsProvider();

    ASSERT_NE(provider, nullptr);
    const PlatformMetrics metrics = provider->sample();
    if (metrics.cpuUtilizationPct.has_value()) {
        EXPECT_GE(*metrics.cpuUtilizationPct, 0.0);
        EXPECT_LE(*metrics.cpuUtilizationPct, 100.0);
    }
    if (metrics.temperatureCelsius.has_value()) {
        EXPECT_TRUE(std::isfinite(*metrics.temperatureCelsius));
    }
}

}  // namespace
}  // namespace snapsentinel::benchmark
