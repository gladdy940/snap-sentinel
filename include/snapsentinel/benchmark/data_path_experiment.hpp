#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace snapsentinel::benchmark {

struct DataPathCounters final {
    std::uint64_t framesProcessed = 0;
    std::uint64_t fullFrameStagingCopies = 0;
    std::uint64_t fullTensorStagingCopies = 0;
    std::uint64_t instrumentedStagingBytesCopied = 0;
    std::uint64_t capturedFramePayloadAllocations = 0;
    std::uint64_t baselineScratchStagingAllocations = 0;
    std::uint64_t tensorPayloadAllocations = 0;
    std::uint64_t preprocessingNanoseconds = 0;
};

class DataPathInstrumentation final {
public:
    void recordFullFrameStagingCopy(std::size_t bytes) noexcept;
    void recordFullTensorStagingCopy(std::size_t bytes) noexcept;
    void recordCapturedFramePayloadAllocation() noexcept;
    void recordBaselineScratchStagingAllocation() noexcept;
    void recordTensorPayloadAllocation() noexcept;
    void recordFrameProcessed() noexcept;
    void recordPreprocessingTime(std::uint64_t nanoseconds) noexcept;
    void reset() noexcept;
    [[nodiscard]] DataPathCounters snapshot() const noexcept;

private:
    std::atomic<std::uint64_t> framesProcessed_{0};
    std::atomic<std::uint64_t> fullFrameStagingCopies_{0};
    std::atomic<std::uint64_t> fullTensorStagingCopies_{0};
    std::atomic<std::uint64_t> instrumentedStagingBytesCopied_{0};
    std::atomic<std::uint64_t> capturedFramePayloadAllocations_{0};
    std::atomic<std::uint64_t> baselineScratchStagingAllocations_{0};
    std::atomic<std::uint64_t> tensorPayloadAllocations_{0};
    std::atomic<std::uint64_t> preprocessingNanoseconds_{0};
};

struct DataPathExperimentOptions final {
    std::size_t frameCount = 1000;
    std::size_t width = 64;
    std::size_t height = 48;
    std::size_t iterations = 3;
    std::size_t warmupFrames = 20;
    std::filesystem::path csvPath{"data_path_experiment.csv"};
};

struct DataPathMetrics final {
    double averageLatencyMs = 0.0;
    double minLatencyMs = 0.0;
    double maxLatencyMs = 0.0;
    double inferenceLatencyMs = 0.0;
    double framesPerSecond = 0.0;
    double elapsedSeconds = 0.0;
};

struct DataPathExperimentResult final {
    DataPathExperimentOptions options;
    DataPathMetrics baselinePerformance;
    DataPathMetrics snapSentinelPerformance;
    DataPathCounters baselineAccounting;
    DataPathCounters snapSentinelAccounting;
};

[[nodiscard]] DataPathExperimentResult runDataPathExperiment(
    const DataPathExperimentOptions& options
);

}  // namespace snapsentinel::benchmark
