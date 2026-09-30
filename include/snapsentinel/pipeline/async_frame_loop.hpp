#pragma once

#include "snapsentinel/pipeline/camera_capture.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>

namespace snapsentinel::pipeline {

class AsyncFrameLoop final {
public:
    using FrameHandler = std::function<void(core::CapturedFrame)>;

    AsyncFrameLoop(
        ICameraCapture& camera,
        FrameHandler handler,
        std::size_t queueCapacity = 4
    );
    ~AsyncFrameLoop();

    AsyncFrameLoop(const AsyncFrameLoop&) = delete;
    AsyncFrameLoop& operator=(const AsyncFrameLoop&) = delete;

    void start();
    void stop();

private:
    void captureFrames(std::stop_token stopToken) noexcept;
    void dispatchFrames() noexcept;
    void rethrowWorkerFailure();
    void storeWorkerFailure(std::exception_ptr failure) noexcept;

    ICameraCapture& camera_;
    FrameHandler handler_;
    const std::size_t queueCapacity_;
    std::jthread captureWorker_;
    std::jthread dispatchWorker_;
    std::stop_source captureStopSource_;
    std::mutex lifecycleMutex_;
    std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    std::deque<core::CapturedFrame> queue_;
    bool captureFinished_ = false;
    std::exception_ptr workerFailure_;
};

}  // namespace snapsentinel::pipeline
