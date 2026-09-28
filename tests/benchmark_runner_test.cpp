#include "snapsentinel/pipeline/benchmark_runner.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>

namespace snapsentinel::pipeline {
namespace {

TEST(BenchmarkOptionsTest, ParsesDurationAndCsvPath) {
    const char* const arguments[] = {
        "snapsentinel_runner",
        "--duration-sec",
        "42",
        "--output-csv",
        "run.csv"
    };

    const BenchmarkOptions options =
        BenchmarkOptions::parseCommandLine(5, arguments);

    EXPECT_EQ(options.duration, std::chrono::seconds(42));
    EXPECT_EQ(options.csvPath, "run.csv");
}

TEST(BenchmarkOptionsTest, RejectsDurationsOverThirtyMinutes) {
    const char* const arguments[] = {
        "snapsentinel_runner",
        "--duration-sec",
        "1801"
    };

    EXPECT_THROW(BenchmarkOptions::parseCommandLine(3, arguments), std::invalid_argument);
}

TEST(BenchmarkOptionsTest, DefaultsToMaximumDurationAndCsvPath) {
    const char* const arguments[] = {"snapsentinel_runner"};

    const BenchmarkOptions options =
        BenchmarkOptions::parseCommandLine(1, arguments);

    EXPECT_EQ(options.duration, std::chrono::minutes(30));
    EXPECT_EQ(options.csvPath, "benchmark.csv");
}

}  // namespace
}  // namespace snapsentinel::pipeline
