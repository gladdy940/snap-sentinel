#pragma once

#include "snapsentinel/pipeline/camera_capture.hpp"

#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace snapsentinel::pipeline {

class AsyncFrameLoop final {
public:
    using FrameHandler = std::function<void(CapturedFrame)>;

    AsyncFrameLoop(ICameraCapture& camera, FrameHandler handler);
    ~AsyncFrameLoop();

    AsyncFrameLoop(const AsyncFrameLoop&) = delete;
    AsyncFrameLoop& operator=(const AsyncFrameLoop&) = delete;

    void start();
    void stop();

private:
    void run(std::stop_token stopToken) noexcept;
    void rethrowWorkerFailure();

    ICameraCapture& camera_;
    FrameHandler handler_;
    std::jthread worker_;
    std::mutex failureMutex_;
    std::exception_ptr workerFailure_;
};

}  // namespace snapsentinel::pipeline
