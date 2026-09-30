#include "snapsentinel/benchmark/benchmark_runner.hpp"

#include "snapsentinel/inference/inference_engine.hpp"
#include "snapsentinel/inference/tensor_preparer.hpp"
#include "snapsentinel/pipeline/camera_capture.hpp"
#include "snapsentinel/pipeline/frame_pipeline.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace snapsentinel::pipeline {
namespace {

using BenchmarkClock = std::chrono::high_resolution_clock;
constexpr auto kSampleInterval = std::chrono::milliseconds(100);
constexpr auto kMaximumDuration = std::chrono::minutes(30);

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

struct PipelineSample final {
    std::uint64_t frameId;
    double latencyMs;
    double inferenceTimeMs;
    double score;
    bool succeeded;
};

class FrameLimitCamera final : public ICameraCapture {
public:
    FrameLimitCamera(ICameraCapture& camera, const std::size_t frameLimit)
        : camera_(camera), frameLimit_(frameLimit) {}

    [[nodiscard]] std::optional<core::CapturedFrame> capture(
        const std::stop_token stopToken
    ) override {
        if (framesCaptured_ >= frameLimit_) {
            return std::nullopt;
        }
        std::optional<core::CapturedFrame> frame = camera_.capture(stopToken);
        if (frame.has_value()) {
            ++framesCaptured_;
        }
        return frame;
    }

private:
    ICameraCapture& camera_;
    const std::size_t frameLimit_;
    std::size_t framesCaptured_ = 0;
};

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

BenchmarkRunner::BenchmarkRunner(
    InferenceSubmitter submitter,
    std::unique_ptr<benchmark::PlatformMetricsProvider> metricsProvider
)
    : submitter_(std::move(submitter)),
      metricsProvider_(
          metricsProvider != nullptr
              ? std::move(metricsProvider)
              : benchmark::createPlatformMetricsProvider()
      ) {
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

BenchmarkSummary BenchmarkRunner::runPipeline(
    ICameraCapture& camera,
    inference::TensorPreparer& tensorPreparer,
    inference::InferenceEngine& inferenceEngine,
    const std::size_t frameCount,
    const std::filesystem::path& csvPath
) {
    if (frameCount == 0) {
        throw std::invalid_argument("Pipeline benchmark frame count must be greater than zero");
    }
    if (csvPath.empty()) {
        throw std::invalid_argument("Pipeline benchmark CSV path must not be empty");
    }

    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("Unable to open benchmark CSV: " + csvPath.string());
    }
    csv << "Frame_ID,Pipeline_Latency_ms,Inference_Time_ms,Score,Succeeded\n";
    csv << std::fixed << std::setprecision(6);

    std::mutex samplesMutex;
    std::condition_variable samplesCondition;
    std::vector<PipelineSample> samples;
    samples.reserve(frameCount);
    FrameLimitCamera limitedCamera(camera, frameCount);
    FramePipeline pipeline(
        limitedCamera,
        tensorPreparer,
        inferenceEngine,
        [&](const std::uint64_t frameId, inference::InferenceResult result) {
            const PipelineSample sample{
                frameId,
                std::chrono::duration<double, std::milli>(result.pipelineLatency).count(),
                std::chrono::duration<double, std::milli>(result.inferenceTime).count(),
                result.score,
                result.succeeded()
            };
            {
                std::lock_guard lock(samplesMutex);
                samples.push_back(sample);
            }
            samplesCondition.notify_all();
        }
    );

    const auto start = std::chrono::steady_clock::now();
    pipeline.start();
    std::unique_lock samplesLock(samplesMutex);
    const bool receivedAllFrames = samplesCondition.wait_for(
        samplesLock,
        std::chrono::seconds(30),
        [&samples, frameCount] { return samples.size() >= frameCount; }
    );
    samplesLock.unlock();

    std::exception_ptr stopFailure;
    try {
        pipeline.stop();
    } catch (...) {
        stopFailure = std::current_exception();
    }
    if (stopFailure != nullptr) {
        std::rethrow_exception(stopFailure);
    }
    if (!receivedAllFrames) {
        throw std::runtime_error(
            "Pipeline benchmark timed out before receiving the requested frame count"
        );
    }
    const double elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    BenchmarkSummary summary;
    summary.completedFrames = static_cast<std::uint64_t>(samples.size());
    summary.elapsedSeconds = elapsedSeconds;
    summary.minLatencyMs = std::numeric_limits<double>::infinity();
    summary.maxLatencyMs = 0.0;
    std::vector<double> latencies;
    latencies.reserve(samples.size());
    double latencyTotal = 0.0;
    double inferenceTotal = 0.0;
    for (const PipelineSample& sample : samples) {
        csv << sample.frameId << ',' << sample.latencyMs << ','
            << sample.inferenceTimeMs << ',' << sample.score << ','
            << (sample.succeeded ? 1 : 0) << '\n';
        latencies.push_back(sample.latencyMs);
        latencyTotal += sample.latencyMs;
        inferenceTotal += sample.inferenceTimeMs;
        summary.minLatencyMs = std::min(summary.minLatencyMs, sample.latencyMs);
        summary.maxLatencyMs = std::max(summary.maxLatencyMs, sample.latencyMs);
        if (!sample.succeeded) {
            ++summary.failedFrames;
        }
    }
    csv.flush();
    if (!csv) {
        throw std::runtime_error("Failed while writing pipeline benchmark CSV");
    }

    if (!samples.empty()) {
        summary.meanLatencyMs = latencyTotal / static_cast<double>(samples.size());
        summary.meanInferenceTimeMs = inferenceTotal / static_cast<double>(samples.size());
        summary.p99LatencyMs = percentile99(latencies);
    } else {
        summary.minLatencyMs = 0.0;
    }
    if (elapsedSeconds > 0.0) {
        summary.framesPerSecond = static_cast<double>(summary.completedFrames) / elapsedSeconds;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "Pipeline benchmark completed frames: " << summary.completedFrames << '\n'
              << "Elapsed time: " << summary.elapsedSeconds << " s\n"
              << "Latency (avg/min/max): " << summary.meanLatencyMs << " / "
              << summary.minLatencyMs << " / " << summary.maxLatencyMs << " ms\n"
              << "Average inference time: " << summary.meanInferenceTimeMs << " ms\n"
              << "Throughput: " << summary.framesPerSecond << " FPS\n"
              << "Failed frames: " << summary.failedFrames << '\n';
    return summary;
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
                const benchmark::PlatformMetrics platformSample = metricsProvider_->sample();
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
                    << completionLatency << ',' << dispatchLatency << ',';
                if (platformSample.cpuUtilizationPct.has_value()) {
                    csv << *platformSample.cpuUtilizationPct;
                }
                csv << ',';
                if (platformSample.temperatureCelsius.has_value()) {
                    csv << *platformSample.temperatureCelsius;
                }
                csv << ',';
                if (platformSample.cpuThrottled.has_value()) {
                    csv << (*platformSample.cpuThrottled ? 1 : 0);
                }
                csv << '\n';
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
