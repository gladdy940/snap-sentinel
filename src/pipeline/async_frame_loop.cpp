#include "snapsentinel/pipeline/async_frame_loop.hpp"

#include <stdexcept>
#include <utility>

namespace snapsentinel::pipeline {

AsyncFrameLoop::AsyncFrameLoop(ICameraCapture& camera, FrameHandler handler)
    : camera_(camera), handler_(std::move(handler)) {
    if (!handler_) {
        throw std::invalid_argument("Frame handler must not be empty");
    }
}

AsyncFrameLoop::~AsyncFrameLoop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
}

void AsyncFrameLoop::start() {
    if (worker_.joinable()) {
        throw std::logic_error("Frame loop is already running");
    }

    {
        std::lock_guard lock(failureMutex_);
        workerFailure_ = nullptr;
    }
    worker_ = std::jthread([this](const std::stop_token stopToken) { run(stopToken); });
}

void AsyncFrameLoop::stop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
    rethrowWorkerFailure();
}

void AsyncFrameLoop::run(const std::stop_token stopToken) noexcept {
    try {
        while (!stopToken.stop_requested()) {
            std::optional<CapturedFrame> frame = camera_.capture(stopToken);
            if (!frame.has_value()) {
                return;
            }
            handler_(std::move(*frame));
        }
    } catch (...) {
        std::lock_guard lock(failureMutex_);
        workerFailure_ = std::current_exception();
    }
}

void AsyncFrameLoop::rethrowWorkerFailure() {
    std::exception_ptr failure;
    {
        std::lock_guard lock(failureMutex_);
        failure = std::exchange(workerFailure_, nullptr);
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

}  // namespace snapsentinel::pipeline
