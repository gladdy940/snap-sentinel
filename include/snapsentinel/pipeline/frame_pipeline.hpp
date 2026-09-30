#pragma once

#include "snapsentinel/inference/inference_engine.hpp"
#include "snapsentinel/inference/tensor_preparer.hpp"
#include "snapsentinel/pipeline/async_frame_loop.hpp"
#include "snapsentinel/pipeline/camera_capture.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <cstddef>

namespace snapsentinel::pipeline {

struct PipelineStatistics final {
    std::uint64_t completedFrames = 0;
    std::uint64_t successfulFrames = 0;
    std::uint64_t failedFrames = 0;
};

class FramePipeline final {
public:
    using CompletionHandler =
        std::function<void(std::uint64_t frameId, inference::InferenceResult result)>;

    FramePipeline(
        inference::TensorPreparer& tensorPreparer,
        inference::InferenceEngine& inferenceEngine,
        CompletionHandler completionHandler
    );

    FramePipeline(
        ICameraCapture& camera,
        inference::TensorPreparer& tensorPreparer,
        inference::InferenceEngine& inferenceEngine,
        CompletionHandler completionHandler,
        std::size_t queueCapacity = 4
    );

    FramePipeline(const FramePipeline&) = delete;
    FramePipeline& operator=(const FramePipeline&) = delete;

    void start();
    void stop();
    void processFrame(core::CapturedFrame frame);
    void submit(std::uint64_t frameId, core::CapturedFrame frame);
    [[nodiscard]] PipelineStatistics statistics() const noexcept;

private:
    struct State;

    std::shared_ptr<State> state_;
    ICameraCapture* camera_ = nullptr;
    std::unique_ptr<AsyncFrameLoop> asyncLoop_;
    std::size_t queueCapacity_ = 4;
};

}  // namespace snapsentinel::pipeline
