#include "snapsentinel/pipeline/async_frame_loop.hpp"

#include <stdexcept>
#include <utility>

namespace snapsentinel::pipeline {

AsyncFrameLoop::AsyncFrameLoop(
    ICameraCapture& camera,
    FrameHandler handler,
    const std::size_t queueCapacity
)
    : camera_(camera),
      handler_(std::move(handler)),
      queueCapacity_(queueCapacity) {
    if (!handler_) {
        throw std::invalid_argument("Frame handler must not be empty");
    }
    if (queueCapacity_ == 0) {
        throw std::invalid_argument("Frame queue capacity must be greater than zero");
    }
}

AsyncFrameLoop::~AsyncFrameLoop() {
    try {
        stop();
    } catch (...) {
    }
}

void AsyncFrameLoop::start() {
    std::lock_guard lifecycleLock(lifecycleMutex_);
    if (captureWorker_.joinable() || dispatchWorker_.joinable()) {
        throw std::logic_error("Frame loop is already running");
    }

    {
        std::lock_guard queueLock(queueMutex_);
        queue_.clear();
        captureFinished_ = false;
        workerFailure_ = nullptr;
    }
    captureStopSource_ = std::stop_source{};

    dispatchWorker_ = std::jthread([this] { dispatchFrames(); });
    try {
        const std::stop_token stopToken = captureStopSource_.get_token();
        captureWorker_ = std::jthread([this, stopToken](std::stop_token) {
            captureFrames(stopToken);
        });
    } catch (...) {
        {
            std::lock_guard queueLock(queueMutex_);
            captureFinished_ = true;
        }
        queueCondition_.notify_all();
        dispatchWorker_.join();
        throw;
    }
}

void AsyncFrameLoop::stop() {
    {
        std::lock_guard lifecycleLock(lifecycleMutex_);
        if (captureWorker_.joinable()) {
            captureStopSource_.request_stop();
            queueCondition_.notify_all();
            captureWorker_.join();
        }
        {
            std::lock_guard queueLock(queueMutex_);
            captureFinished_ = true;
        }
        queueCondition_.notify_all();
        if (dispatchWorker_.joinable()) {
            dispatchWorker_.join();
        }
    }
    rethrowWorkerFailure();
}

void AsyncFrameLoop::captureFrames(const std::stop_token stopToken) noexcept {
    try {
        while (!stopToken.stop_requested()) {
            std::optional<core::CapturedFrame> frame = camera_.capture(stopToken);
            if (!frame.has_value()) {
                break;
            }

            std::unique_lock queueLock(queueMutex_);
            queueCondition_.wait(queueLock, [this, stopToken] {
                return queue_.size() < queueCapacity_ || stopToken.stop_requested();
            });
            if (stopToken.stop_requested()) {
                break;
            }
            queue_.push_back(std::move(*frame));
            queueLock.unlock();
            queueCondition_.notify_all();
        }
    } catch (...) {
        storeWorkerFailure(std::current_exception());
    }

    {
        std::lock_guard queueLock(queueMutex_);
        captureFinished_ = true;
    }
    queueCondition_.notify_all();
}

void AsyncFrameLoop::dispatchFrames() noexcept {
    while (true) {
        core::CapturedFrame frame;
        {
            std::unique_lock queueLock(queueMutex_);
            queueCondition_.wait(queueLock, [this] {
                return !queue_.empty() || captureFinished_;
            });
            if (queue_.empty()) {
                return;
            }
            frame = std::move(queue_.front());
            queue_.pop_front();
        }
        queueCondition_.notify_all();

        try {
            handler_(std::move(frame));
        } catch (...) {
            storeWorkerFailure(std::current_exception());
            captureStopSource_.request_stop();
            {
                std::lock_guard queueLock(queueMutex_);
                queue_.clear();
                captureFinished_ = true;
            }
            queueCondition_.notify_all();
            return;
        }
    }
}

void AsyncFrameLoop::storeWorkerFailure(std::exception_ptr failure) noexcept {
    std::lock_guard queueLock(queueMutex_);
    if (workerFailure_ == nullptr) {
        workerFailure_ = std::move(failure);
    }
}

void AsyncFrameLoop::rethrowWorkerFailure() {
    std::exception_ptr failure;
    {
        std::lock_guard queueLock(queueMutex_);
        failure = std::exchange(workerFailure_, nullptr);
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

}  // namespace snapsentinel::pipeline
