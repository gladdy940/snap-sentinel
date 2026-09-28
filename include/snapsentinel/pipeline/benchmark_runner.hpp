#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace snapsentinel::pipeline {

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
};

class BenchmarkRunner final {
public:
    using CompletionCallback = std::function<void()>;
    using InferenceSubmitter =
        std::function<void(std::uint64_t frameId, CompletionCallback completion)>;

    // The submitter must invoke completion exactly once for each accepted request.
    // Tensor storage captured by the submitter must remain valid until completion.
    explicit BenchmarkRunner(InferenceSubmitter submitter);

    [[nodiscard]] static BenchmarkSummary runCommandLine(
        int argc,
        const char* const argv[],
        InferenceSubmitter submitter
    );
    [[nodiscard]] BenchmarkSummary run(const BenchmarkOptions& options) const;

private:
    InferenceSubmitter submitter_;
};

}  // namespace snapsentinel::pipeline
