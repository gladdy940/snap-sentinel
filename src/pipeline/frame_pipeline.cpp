#include "snapsentinel/pipeline/frame_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace snapsentinel::pipeline {

struct FramePipeline::State final {
    State(
        inference::TensorPreparer& tensorPreparer,
        inference::InferenceEngine& inferenceEngine,
        CompletionHandler completionHandler
    )
        : tensorPreparer(tensorPreparer),
          inferenceEngine(inferenceEngine),
          completionHandler(std::move(completionHandler)) {}

    inference::TensorPreparer& tensorPreparer;
    inference::InferenceEngine& inferenceEngine;
    CompletionHandler completionHandler;
    std::atomic<std::uint64_t> completedFrames{0};
    std::atomic<std::uint64_t> successfulFrames{0};
    std::atomic<std::uint64_t> failedFrames{0};
};

FramePipeline::FramePipeline(
    inference::TensorPreparer& tensorPreparer,
    inference::InferenceEngine& inferenceEngine,
    CompletionHandler completionHandler
)
    : state_(std::make_shared<State>(
          tensorPreparer,
          inferenceEngine,
          std::move(completionHandler)
      )) {
    if (!state_->completionHandler) {
        throw std::invalid_argument("Frame completion handler must not be empty");
    }
}

FramePipeline::FramePipeline(
    ICameraCapture& camera,
    inference::TensorPreparer& tensorPreparer,
    inference::InferenceEngine& inferenceEngine,
    CompletionHandler completionHandler,
    const std::size_t queueCapacity
)
    : state_(std::make_shared<State>(
          tensorPreparer,
          inferenceEngine,
          std::move(completionHandler)
      )),
      camera_(&camera),
      queueCapacity_(queueCapacity) {
    if (!state_->completionHandler) {
        throw std::invalid_argument("Frame completion handler must not be empty");
    }
}

void FramePipeline::start() {
    if (camera_ == nullptr) {
        throw std::logic_error("Frame pipeline has no camera capture source");
    }
    if (asyncLoop_ != nullptr) {
        throw std::logic_error("Frame pipeline is already running");
    }
    auto loop = std::make_unique<AsyncFrameLoop>(
        *camera_,
        [this](core::CapturedFrame frame) {
            processFrame(std::move(frame));
        },
        queueCapacity_
    );
    loop->start();
    asyncLoop_ = std::move(loop);
}

void FramePipeline::stop() {
    if (asyncLoop_ != nullptr) {
        std::unique_ptr<AsyncFrameLoop> loop = std::move(asyncLoop_);
        loop->stop();
    }
}

void FramePipeline::processFrame(core::CapturedFrame frame) {
    const std::uint64_t frameId = frame.id;
    submit(frameId, std::move(frame));
}

void FramePipeline::submit(
    const std::uint64_t frameId,
    core::CapturedFrame frame
) {
    const core::CapturedFrame::Clock::time_point captureTime = frame.timestamp;
    std::unique_ptr<inference::PreparedInference> prepared =
        state_->tensorPreparer.prepare(frame);
    if (prepared == nullptr) {
        throw std::runtime_error("Tensor preparer returned an empty inference request");
    }

    auto request = std::shared_ptr<inference::PreparedInference>(std::move(prepared));
    state_->inferenceEngine.submit(
        *request,
        [state = state_, request, frameId, captureTime](
            inference::InferenceResult result
        ) {
            static_cast<void>(request);
            result.pipelineLatency = std::chrono::duration_cast<std::chrono::nanoseconds>(
                core::CapturedFrame::Clock::now() - captureTime
            );
            state->completedFrames.fetch_add(1, std::memory_order_relaxed);
            if (result.succeeded()) {
                state->successfulFrames.fetch_add(1, std::memory_order_relaxed);
            } else {
                state->failedFrames.fetch_add(1, std::memory_order_relaxed);
            }
            state->completionHandler(frameId, std::move(result));
        }
    );
}

PipelineStatistics FramePipeline::statistics() const noexcept {
    return PipelineStatistics{
        state_->completedFrames.load(std::memory_order_relaxed),
        state_->successfulFrames.load(std::memory_order_relaxed),
        state_->failedFrames.load(std::memory_order_relaxed)
    };
}

}  // namespace snapsentinel::pipeline
