#include "snapsentinel/benchmark/data_path_experiment.hpp"

#include "snapsentinel/core/frame.hpp"
#include "snapsentinel/inference/inference_engine.hpp"
#include "snapsentinel/inference/tensor_preparer.hpp"
#include "snapsentinel/pipeline/camera_capture.hpp"
#include "snapsentinel/pipeline/frame_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifndef SNAP_SENTINEL_BUILD_CONFIGURATION
#define SNAP_SENTINEL_BUILD_CONFIGURATION "unspecified"
#endif

namespace snapsentinel::benchmark {
namespace {

using Clock = std::chrono::steady_clock;

template <bool Instrument>
class InstrumentedCameraCapture final : public pipeline::ICameraCapture {
public:
    InstrumentedCameraCapture(
        pipeline::ICameraCapture& camera,
        DataPathInstrumentation* instrumentation
    ) noexcept
        : camera_(camera), instrumentation_(instrumentation) {}

    [[nodiscard]] std::optional<core::CapturedFrame> capture(
        const std::stop_token stopToken
    ) override {
        std::optional<core::CapturedFrame> frame = camera_.capture(stopToken);
        if constexpr (Instrument) {
            if (frame.has_value() && frame->buffer != nullptr) {
                instrumentation_->recordCapturedFramePayloadAllocation();
            }
        }
        return frame;
    }

private:
    pipeline::ICameraCapture& camera_;
    DataPathInstrumentation* instrumentation_;
};

template <bool Instrument>
class ConventionalBaselinePreparer final : public inference::TensorPreparer {
public:
    explicit ConventionalBaselinePreparer(
        DataPathInstrumentation* instrumentation
    ) noexcept
        : instrumentation_(instrumentation) {}

    [[nodiscard]] std::unique_ptr<inference::PreparedInference> prepare(
        const core::CapturedFrame& frame
    ) override {
        const auto start = [] {
            if constexpr (Instrument) {
                return Clock::now();
            } else {
                return Clock::time_point{};
            }
        }();
        if (frame.pixelFormat != core::PixelFormat::gray8 || frame.buffer == nullptr) {
            throw std::invalid_argument("Baseline preparer requires a buffered gray8 frame");
        }
        const std::size_t pixelCount = checkedPixelCount(frame.width, frame.height);
        const std::span<const std::byte> pixels = frame.pixels();
        if (pixels.size() != pixelCount) {
            throw std::invalid_argument("Baseline frame storage does not match its dimensions");
        }
        if (pixelCount > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("Baseline tensor size overflows size_t");
        }

        std::vector<std::byte> preprocessingPixels(pixelCount);
        if constexpr (Instrument) {
            instrumentation_->recordBaselineScratchStagingAllocation();
        }
        std::copy(pixels.begin(), pixels.end(), preprocessingPixels.begin());
        if constexpr (Instrument) {
            instrumentation_->recordFullFrameStagingCopy(pixelCount);
        }

        std::vector<float> preprocessedTensor(pixelCount);
        if constexpr (Instrument) {
            instrumentation_->recordBaselineScratchStagingAllocation();
        }
        for (std::size_t index = 0; index < pixelCount; ++index) {
            preprocessedTensor[index] =
                static_cast<float>(
                    std::to_integer<unsigned int>(preprocessingPixels[index])
                ) / 255.0F;
        }

        inference::Tensor input;
        input.shape = {1, frame.height, frame.width, 1};
        input.elementType = inference::TensorElementType::float32;
        input.layout = inference::TensorLayout::nhwc;
        input.storage = core::Buffer(pixelCount * sizeof(float));
        if constexpr (Instrument) {
            instrumentation_->recordTensorPayloadAllocation();
        }
        std::memcpy(
            input.storage.data(),
            preprocessedTensor.data(),
            pixelCount * sizeof(float)
        );
        if constexpr (Instrument) {
            instrumentation_->recordFullTensorStagingCopy(pixelCount * sizeof(float));
        }

        auto batch = std::make_unique<inference::TensorBatch>();
        batch->inputs.push_back(std::move(input));
        if constexpr (Instrument) {
            recordElapsed(start);
        }
        return batch;
    }

private:
    static std::size_t checkedPixelCount(
        const std::size_t width,
        const std::size_t height
    ) {
        if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
            throw std::overflow_error("Baseline frame dimensions overflow size_t");
        }
        return width * height;
    }

    void recordElapsed(const Clock::time_point start) noexcept {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start
        ).count();
        instrumentation_->recordPreprocessingTime(static_cast<std::uint64_t>(elapsed));
    }

    DataPathInstrumentation* instrumentation_;
};

template <bool Instrument>
class LowCopyPreparer final : public inference::TensorPreparer {
public:
    explicit LowCopyPreparer(DataPathInstrumentation* instrumentation) noexcept
        : instrumentation_(instrumentation) {}

    [[nodiscard]] std::unique_ptr<inference::PreparedInference> prepare(
        const core::CapturedFrame& frame
    ) override {
        const auto start = [] {
            if constexpr (Instrument) {
                return Clock::now();
            } else {
                return Clock::time_point{};
            }
        }();
        std::unique_ptr<inference::PreparedInference> request = preparer_.prepare(frame);
        const auto* const batch =
            dynamic_cast<const inference::TensorBatch*>(request.get());
        if (batch == nullptr || batch->inputs.empty()) {
            throw std::runtime_error("Low-copy preparer did not produce an input tensor");
        }
        if constexpr (Instrument) {
            instrumentation_->recordTensorPayloadAllocation();
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start
            ).count();
            instrumentation_->recordPreprocessingTime(static_cast<std::uint64_t>(elapsed));
        }
        return request;
    }

private:
    inference::DeterministicTensorPreparer preparer_;
    DataPathInstrumentation* instrumentation_;
};

struct FrameSample final {
    std::uint64_t frameId;
    double pipelineLatencyMs;
    double inferenceLatencyMs;
    double score;
    bool succeeded;
};

struct PassResult final {
    std::vector<FrameSample> samples;
    DataPathCounters counters;
    double elapsedSeconds = 0.0;
};

template <bool Instrument>
PassResult runPass(
    const std::size_t width,
    const std::size_t height,
    const std::size_t frameCount,
    const bool useBaseline
) {
    if (width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Frame dimensions exceed the mock camera limits");
    }

    DataPathInstrumentation instrumentation;
    DataPathInstrumentation* const instrumentationPointer =
        Instrument ? &instrumentation : nullptr;
    pipeline::MockCameraCapture source(
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
        frameCount
    );
    InstrumentedCameraCapture<Instrument> camera(source, instrumentationPointer);
    ConventionalBaselinePreparer<Instrument> baselinePreparer(instrumentationPointer);
    LowCopyPreparer<Instrument> lowCopyPreparer(instrumentationPointer);
    inference::MockInferenceEngine inferenceEngine;
    std::mutex samplesMutex;
    std::condition_variable samplesCondition;
    PassResult result;
    result.samples.reserve(frameCount);

    pipeline::FramePipeline framePipeline(
        camera,
        useBaseline
            ? static_cast<inference::TensorPreparer&>(baselinePreparer)
            : static_cast<inference::TensorPreparer&>(lowCopyPreparer),
        inferenceEngine,
        [&result, &instrumentation, &samplesMutex, &samplesCondition](
            const std::uint64_t frameId,
            inference::InferenceResult inferenceResult
        ) {
            const FrameSample sample{
                frameId,
                std::chrono::duration<double, std::milli>(
                    inferenceResult.pipelineLatency
                ).count(),
                std::chrono::duration<double, std::milli>(
                    inferenceResult.inferenceTime
                ).count(),
                inferenceResult.score,
                inferenceResult.succeeded()
            };
            {
                std::lock_guard lock(samplesMutex);
                result.samples.push_back(sample);
                if constexpr (Instrument) {
                    instrumentation.recordFrameProcessed();
                }
            }
            samplesCondition.notify_all();
        }
    );

    const auto start = Clock::now();
    framePipeline.start();
    std::unique_lock lock(samplesMutex);
    const bool receivedAllFrames = samplesCondition.wait_for(
        lock,
        std::chrono::seconds(60),
        [&result, frameCount] { return result.samples.size() >= frameCount; }
    );
    lock.unlock();

    std::exception_ptr stopFailure;
    try {
        framePipeline.stop();
    } catch (...) {
        stopFailure = std::current_exception();
    }
    if (stopFailure != nullptr) {
        std::rethrow_exception(stopFailure);
    }
    if (!receivedAllFrames) {
        throw std::runtime_error("Data-path experiment timed out waiting for frames");
    }

    result.elapsedSeconds = std::chrono::duration<double>(Clock::now() - start).count();
    if constexpr (Instrument) {
        result.counters = instrumentation.snapshot();
    }
    std::sort(
        result.samples.begin(),
        result.samples.end(),
        [](const FrameSample& left, const FrameSample& right) {
            return left.frameId < right.frameId;
        }
    );
    for (const FrameSample& sample : result.samples) {
        if (!sample.succeeded) {
            throw std::runtime_error("Inference failed during data-path experiment");
        }
    }
    return result;
}

class Aggregate final {
public:
    void add(const PassResult& pass) {
        elapsedSeconds_ += pass.elapsedSeconds;
        for (const FrameSample& sample : pass.samples) {
            latenciesMs_.push_back(sample.pipelineLatencyMs);
            inferenceLatenciesMs_.push_back(sample.inferenceLatencyMs);
        }
    }
    [[nodiscard]] DataPathMetrics metrics() const {
        DataPathMetrics result;
        result.elapsedSeconds = elapsedSeconds_;
        if (!latenciesMs_.empty()) {
            result.averageLatencyMs =
                std::accumulate(latenciesMs_.begin(), latenciesMs_.end(), 0.0) /
                static_cast<double>(latenciesMs_.size());
            const auto [minimum, maximum] =
                std::minmax_element(latenciesMs_.begin(), latenciesMs_.end());
            result.minLatencyMs = *minimum;
            result.maxLatencyMs = *maximum;
            result.inferenceLatencyMs =
                std::accumulate(
                    inferenceLatenciesMs_.begin(),
                    inferenceLatenciesMs_.end(),
                    0.0
                ) / static_cast<double>(inferenceLatenciesMs_.size());
            if (elapsedSeconds_ > 0.0) {
                result.framesPerSecond =
                    static_cast<double>(latenciesMs_.size()) / elapsedSeconds_;
            }
        }
        return result;
    }

private:
    std::vector<double> latenciesMs_;
    std::vector<double> inferenceLatenciesMs_;
    double elapsedSeconds_ = 0.0;
};

void accumulateCounters(DataPathCounters& total, const DataPathCounters& sample) {
    total.framesProcessed += sample.framesProcessed;
    total.fullFrameStagingCopies += sample.fullFrameStagingCopies;
    total.fullTensorStagingCopies += sample.fullTensorStagingCopies;
    total.instrumentedStagingBytesCopied += sample.instrumentedStagingBytesCopied;
    total.capturedFramePayloadAllocations += sample.capturedFramePayloadAllocations;
    total.baselineScratchStagingAllocations += sample.baselineScratchStagingAllocations;
    total.tensorPayloadAllocations += sample.tensorPayloadAllocations;
    total.preprocessingNanoseconds += sample.preprocessingNanoseconds;
}

void validateOutputEquivalence(const PassResult& baseline, const PassResult& lowCopy) {
    if (baseline.samples.size() != lowCopy.samples.size()) {
        throw std::runtime_error("Data-path experiments processed different frame counts");
    }
    for (std::size_t index = 0; index < baseline.samples.size(); ++index) {
        if (baseline.samples[index].frameId != lowCopy.samples[index].frameId ||
            baseline.samples[index].score != lowCopy.samples[index].score ||
            baseline.samples[index].succeeded != lowCopy.samples[index].succeeded) {
            throw std::runtime_error(
                "Baseline and Snap-Sentinel produced different logical inference results"
            );
        }
    }
}

void writeMetrics(
    std::ofstream& csv,
    const std::string_view pipelineName,
    const DataPathExperimentOptions& options,
    const std::string_view measurementMode,
    const std::size_t iteration,
    const DataPathMetrics* performance,
    const DataPathCounters* accounting
) {
    csv << measurementMode << ',' << SNAP_SENTINEL_BUILD_CONFIGURATION << ','
        << pipelineName << ',' << iteration << ',' << options.frameCount << ','
        << options.width << ',' << options.height << ',' << options.warmupFrames << ',';
    const auto writePerformanceValue = [&csv](const double* value) {
        if (value != nullptr) {
            csv << *value;
        }
        csv << ',';
    };
    writePerformanceValue(performance != nullptr ? &performance->averageLatencyMs : nullptr);
    writePerformanceValue(performance != nullptr ? &performance->minLatencyMs : nullptr);
    writePerformanceValue(performance != nullptr ? &performance->maxLatencyMs : nullptr);
    writePerformanceValue(performance != nullptr ? &performance->framesPerSecond : nullptr);
    writePerformanceValue(performance != nullptr ? &performance->inferenceLatencyMs : nullptr);
    if (accounting != nullptr) {
        csv << accounting->framesProcessed << ',' << accounting->fullFrameStagingCopies << ','
            << accounting->fullTensorStagingCopies << ','
            << accounting->instrumentedStagingBytesCopied << ','
            << accounting->capturedFramePayloadAllocations << ','
            << accounting->baselineScratchStagingAllocations << ','
            << accounting->tensorPayloadAllocations;
    } else {
        csv << ",,,,,,";
    }
    csv << '\n';
}

void printMetrics(
    const std::string_view name,
    const DataPathMetrics& performance,
    const DataPathCounters& accounting
) {
    std::cout << name << '\n'
              << "--------\n"
              << "Performance measurements:\n"
              << "Average latency: " << performance.averageLatencyMs << " ms\n"
              << "Min latency: " << performance.minLatencyMs << " ms\n"
              << "Max latency: " << performance.maxLatencyMs << " ms\n"
              << "Inference latency (mean): " << performance.inferenceLatencyMs << " ms\n"
              << "FPS: " << performance.framesPerSecond << '\n'
              << "Pipeline latency boundary: frame timestamp -> completion\n"
              << "FPS boundary: wall-clock start -> all completions and worker shutdown\n"
              << "Accounting measurements:\n"
              << "Full-frame staging copies: " << accounting.fullFrameStagingCopies << '\n'
              << "Full-tensor staging copies: " << accounting.fullTensorStagingCopies << '\n'
              << "Instrumented staging bytes copied: "
              << accounting.instrumentedStagingBytesCopied << '\n'
              << "Captured-frame allocations: "
              << accounting.capturedFramePayloadAllocations << '\n'
              << "Scratch/staging allocations: "
              << accounting.baselineScratchStagingAllocations << '\n'
              << "Tensor allocations: " << accounting.tensorPayloadAllocations << '\n';
}

}  // namespace

void DataPathInstrumentation::recordFullFrameStagingCopy(
    const std::size_t bytes
) noexcept {
    fullFrameStagingCopies_.fetch_add(1, std::memory_order_relaxed);
    instrumentedStagingBytesCopied_.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed
    );
}

void DataPathInstrumentation::recordFullTensorStagingCopy(
    const std::size_t bytes
) noexcept {
    fullTensorStagingCopies_.fetch_add(1, std::memory_order_relaxed);
    instrumentedStagingBytesCopied_.fetch_add(
        static_cast<std::uint64_t>(bytes),
        std::memory_order_relaxed
    );
}

void DataPathInstrumentation::recordCapturedFramePayloadAllocation() noexcept {
    capturedFramePayloadAllocations_.fetch_add(1, std::memory_order_relaxed);
}

void DataPathInstrumentation::recordBaselineScratchStagingAllocation() noexcept {
    baselineScratchStagingAllocations_.fetch_add(1, std::memory_order_relaxed);
}

void DataPathInstrumentation::recordTensorPayloadAllocation() noexcept {
    tensorPayloadAllocations_.fetch_add(1, std::memory_order_relaxed);
}

void DataPathInstrumentation::recordFrameProcessed() noexcept {
    framesProcessed_.fetch_add(1, std::memory_order_relaxed);
}

void DataPathInstrumentation::recordPreprocessingTime(
    const std::uint64_t nanoseconds
) noexcept {
    preprocessingNanoseconds_.fetch_add(nanoseconds, std::memory_order_relaxed);
}

void DataPathInstrumentation::reset() noexcept {
    framesProcessed_.store(0, std::memory_order_relaxed);
    fullFrameStagingCopies_.store(0, std::memory_order_relaxed);
    fullTensorStagingCopies_.store(0, std::memory_order_relaxed);
    instrumentedStagingBytesCopied_.store(0, std::memory_order_relaxed);
    capturedFramePayloadAllocations_.store(0, std::memory_order_relaxed);
    baselineScratchStagingAllocations_.store(0, std::memory_order_relaxed);
    tensorPayloadAllocations_.store(0, std::memory_order_relaxed);
    preprocessingNanoseconds_.store(0, std::memory_order_relaxed);
}

DataPathCounters DataPathInstrumentation::snapshot() const noexcept {
    return DataPathCounters{
        framesProcessed_.load(std::memory_order_relaxed),
        fullFrameStagingCopies_.load(std::memory_order_relaxed),
        fullTensorStagingCopies_.load(std::memory_order_relaxed),
        instrumentedStagingBytesCopied_.load(std::memory_order_relaxed),
        capturedFramePayloadAllocations_.load(std::memory_order_relaxed),
        baselineScratchStagingAllocations_.load(std::memory_order_relaxed),
        tensorPayloadAllocations_.load(std::memory_order_relaxed),
        preprocessingNanoseconds_.load(std::memory_order_relaxed)
    };
}

DataPathExperimentResult runDataPathExperiment(
    const DataPathExperimentOptions& options
) {
    if (options.frameCount == 0 || options.iterations == 0 ||
        options.width == 0 || options.height == 0 || options.csvPath.empty()) {
        throw std::invalid_argument("Experiment frames, iterations, dimensions, and CSV path must be valid");
    }
    if (options.iterations >
        std::numeric_limits<std::uint64_t>::max() / options.frameCount) {
        throw std::overflow_error("Experiment total frame count overflows uint64_t");
    }
    if (options.frameCount > std::numeric_limits<std::uint64_t>::max() -
                                 options.warmupFrames) {
        throw std::overflow_error("Experiment warm-up frame count overflows uint64_t");
    }

    std::ofstream csv(options.csvPath, std::ios::out | std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("Unable to open experiment CSV: " + options.csvPath.string());
    }
    csv << "measurement_mode,build_configuration,pipeline,iteration,frames,width,height,"
           "warmup_frames,average_latency_ms,min_latency_ms,max_latency_ms,fps,"
           "inference_latency_ms,frames_processed,full_frame_staging_copies,"
           "full_tensor_staging_copies,instrumented_staging_bytes,"
           "captured_frame_allocations,scratch_staging_allocations,tensor_allocations\n";
    csv << std::fixed << std::setprecision(6);

    if (options.warmupFrames != 0) {
        static_cast<void>(runPass<true>(
            options.width,
            options.height,
            options.warmupFrames,
            true
        ));
        static_cast<void>(runPass<true>(
            options.width,
            options.height,
            options.warmupFrames,
            false
        ));
        static_cast<void>(runPass<false>(
            options.width,
            options.height,
            options.warmupFrames,
            false
        ));
        static_cast<void>(runPass<false>(
            options.width,
            options.height,
            options.warmupFrames,
            true
        ));
    }

    Aggregate baselineAggregate;
    Aggregate lowCopyAggregate;
    DataPathCounters baselineAccounting;
    DataPathCounters lowCopyAccounting;
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
        PassResult baselineAccountingPass;
        PassResult lowCopyAccountingPass;
        PassResult baselinePerformancePass;
        PassResult lowCopyPerformancePass;
        if (iteration % 2 == 0) {
            baselineAccountingPass = runPass<true>(
                options.width,
                options.height,
                options.frameCount,
                true
            );
            lowCopyAccountingPass = runPass<true>(
                options.width,
                options.height,
                options.frameCount,
                false
            );
            baselinePerformancePass = runPass<false>(
                options.width,
                options.height,
                options.frameCount,
                true
            );
            lowCopyPerformancePass = runPass<false>(
                options.width,
                options.height,
                options.frameCount,
                false
            );
        } else {
            lowCopyAccountingPass = runPass<true>(
                options.width,
                options.height,
                options.frameCount,
                false
            );
            baselineAccountingPass = runPass<true>(
                options.width,
                options.height,
                options.frameCount,
                true
            );
            lowCopyPerformancePass = runPass<false>(
                options.width,
                options.height,
                options.frameCount,
                false
            );
            baselinePerformancePass = runPass<false>(
                options.width,
                options.height,
                options.frameCount,
                true
            );
        }
        validateOutputEquivalence(baselineAccountingPass, lowCopyAccountingPass);
        validateOutputEquivalence(baselinePerformancePass, lowCopyPerformancePass);
        validateOutputEquivalence(baselineAccountingPass, baselinePerformancePass);
        validateOutputEquivalence(lowCopyAccountingPass, lowCopyPerformancePass);

        accumulateCounters(baselineAccounting, baselineAccountingPass.counters);
        accumulateCounters(lowCopyAccounting, lowCopyAccountingPass.counters);
        baselineAggregate.add(baselinePerformancePass);
        lowCopyAggregate.add(lowCopyPerformancePass);

        const std::size_t csvIteration = iteration + 1;
        const auto writeAccountingPass = [&](const std::string_view name,
                                             const PassResult& pass) {
            writeMetrics(
                csv,
                name,
                options,
                "accounting",
                csvIteration,
                nullptr,
                &pass.counters
            );
        };
        const auto writePerformancePass = [&](const std::string_view name,
                                              const PassResult& pass) {
            Aggregate oneRun;
            oneRun.add(pass);
            const DataPathMetrics metrics = oneRun.metrics();
            writeMetrics(
                csv,
                name,
                options,
                "performance",
                csvIteration,
                &metrics,
                nullptr
            );
        };
        if (iteration % 2 == 0) {
            writeAccountingPass("baseline", baselineAccountingPass);
            writeAccountingPass("snap-sentinel", lowCopyAccountingPass);
            writePerformancePass("baseline", baselinePerformancePass);
            writePerformancePass("snap-sentinel", lowCopyPerformancePass);
        } else {
            writeAccountingPass("snap-sentinel", lowCopyAccountingPass);
            writeAccountingPass("baseline", baselineAccountingPass);
            writePerformancePass("snap-sentinel", lowCopyPerformancePass);
            writePerformancePass("baseline", baselinePerformancePass);
        }
    }

    DataPathExperimentResult result{
        options,
        baselineAggregate.metrics(),
        lowCopyAggregate.metrics(),
        baselineAccounting,
        lowCopyAccounting
    };

    std::cout << "==================================================\n"
              << "Snap-Sentinel Data-Path Experiment\n"
              << "==================================================\n"
              << "Host mock software architecture experiment; not Snapdragon performance.\n"
              << "Frames per run: " << options.frameCount << '\n'
              << "Resolution: " << options.width << 'x' << options.height << " gray8\n"
              << "Iterations: " << options.iterations << '\n'
              << "Warm-up frames per pipeline: " << options.warmupFrames << '\n'
              << "Datatype: normalized float32 NHWC\n"
              << "Build configuration: " << SNAP_SENTINEL_BUILD_CONFIGURATION << '\n'
              << "Snap-Sentinel per-element tensor memcpy stores and mock-inference "
                 "per-element memcpy reads remain in the algorithms; neither is counted as "
                 "a full-payload staging copy.\n"
              << "Captured-frame allocation/fill occurs before the frame timestamp.\n\n";
    printMetrics("BASELINE", result.baselinePerformance, result.baselineAccounting);
    std::cout << '\n';
    printMetrics(
        "SNAP-SENTINEL",
        result.snapSentinelPerformance,
        result.snapSentinelAccounting
    );
    std::cout << "\n==================================================\nPERFORMANCE DIFFERENCE "
                 "(Snap-Sentinel - Baseline)\n"
              << "==================================================\n"
              << "Latency difference: "
              << result.snapSentinelPerformance.averageLatencyMs -
                     result.baselinePerformance.averageLatencyMs
              << " ms\n"
              << "Throughput difference: "
              << result.snapSentinelPerformance.framesPerSecond -
                     result.baselinePerformance.framesPerSecond
              << " FPS\n\n"
              << "==================================================\nDATA-PATH ACCOUNTING DIFFERENCE "
                 "(Snap-Sentinel - Baseline)\n"
              << "==================================================\n"
              << "Full-frame staging-copy difference: "
              << static_cast<std::int64_t>(
                     result.snapSentinelAccounting.fullFrameStagingCopies
                 ) -
                     static_cast<std::int64_t>(
                         result.baselineAccounting.fullFrameStagingCopies
                     )
              << '\n'
              << "Full-tensor staging-copy difference: "
              << static_cast<std::int64_t>(
                     result.snapSentinelAccounting.fullTensorStagingCopies
                 ) -
                     static_cast<std::int64_t>(
                         result.baselineAccounting.fullTensorStagingCopies
                     )
              << '\n'
              << "Instrumented staging-byte difference: "
              << static_cast<std::int64_t>(
                     result.snapSentinelAccounting.instrumentedStagingBytesCopied
                 ) -
                     static_cast<std::int64_t>(
                         result.baselineAccounting.instrumentedStagingBytesCopied
                     )
              << " bytes\n"
              << "Scratch/staging-allocation difference: "
              << static_cast<std::int64_t>(
                     result.snapSentinelAccounting.baselineScratchStagingAllocations
                 ) -
                     static_cast<std::int64_t>(
                         result.baselineAccounting.baselineScratchStagingAllocations
                     )
              << '\n'
              << "Tensor-payload-allocation difference: "
              << static_cast<std::int64_t>(
                     result.snapSentinelAccounting.tensorPayloadAllocations
                 ) -
                     static_cast<std::int64_t>(
                         result.baselineAccounting.tensorPayloadAllocations
                     )
              << '\n';

    csv.flush();
    if (!csv) {
        throw std::runtime_error("Failed while writing data-path experiment CSV");
    }
    std::cout << "CSV written to " << options.csvPath << '\n';
    return result;
}

}  // namespace snapsentinel::benchmark
