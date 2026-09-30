#pragma once

#include "snapsentinel/benchmark/platform_metrics.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace snapsentinel::inference {
class TensorPreparer;
class InferenceEngine;
}

namespace snapsentinel::pipeline {

class ICameraCapture;

struct BenchmarkOptions final {
    std::chrono::seconds duration{1800};
    std::filesystem::path csvPath{"benchmark.csv"};

    [[nodiscard]] static BenchmarkOptions parseCommandLine(
        int argc,
        const char* const argv[]
    );
    [[nodiscard]] static std::string usage(const char* executableName);
};

struct BenchmarkSummary final {
    std::uint64_t completedFrames = 0;
    double meanLatencyMs = 0.0;
    double p99LatencyMs = 0.0;
    double maxThermalShiftC = 0.0;
    double minLatencyMs = 0.0;
    double maxLatencyMs = 0.0;
    double meanInferenceTimeMs = 0.0;
    double framesPerSecond = 0.0;
    double elapsedSeconds = 0.0;
    std::uint64_t failedFrames = 0;
};

class BenchmarkRunner final {
public:
    using CompletionCallback = std::function<void()>;
    using InferenceSubmitter =
        std::function<void(std::uint64_t frameId, CompletionCallback completion)>;

    // The submitter must invoke completion exactly once for each accepted request.
    // Tensor storage captured by the submitter must remain valid until completion.
    explicit BenchmarkRunner(
        InferenceSubmitter submitter,
        std::unique_ptr<benchmark::PlatformMetricsProvider> metricsProvider = {}
    );

    [[nodiscard]] static BenchmarkSummary runCommandLine(
        int argc,
        const char* const argv[],
        InferenceSubmitter submitter
    );
    [[nodiscard]] static BenchmarkSummary runPipeline(
        ICameraCapture& camera,
        inference::TensorPreparer& tensorPreparer,
        inference::InferenceEngine& inferenceEngine,
        std::size_t frameCount,
        const std::filesystem::path& csvPath
    );
    [[nodiscard]] BenchmarkSummary run(const BenchmarkOptions& options) const;

private:
    InferenceSubmitter submitter_;
    std::unique_ptr<benchmark::PlatformMetricsProvider> metricsProvider_;
};

}  // namespace snapsentinel::pipeline
