#include "snapsentinel/pipeline/camera_capture.hpp"

#include <limits>
#include <stdexcept>

namespace snapsentinel::pipeline {

MockCameraCapture::MockCameraCapture(
    const std::uint32_t width,
    const std::uint32_t height,
    const std::size_t frameCount,
    const std::uint64_t firstFrameId
)
    : width_(width),
      height_(height),
      frameCount_(frameCount),
      firstFrameId_(firstFrameId) {
    if (width_ == 0 || height_ == 0) {
        throw std::invalid_argument("Mock camera dimensions must be greater than zero");
    }
    if (static_cast<std::size_t>(width_) >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height_)) {
        throw std::overflow_error("Mock camera frame dimensions are too large");
    }
    if (frameCount_ != 0 &&
        frameCount_ - 1 > std::numeric_limits<std::uint64_t>::max() - firstFrameId_) {
        throw std::overflow_error("Mock camera frame ID range overflows");
    }
}

std::optional<core::CapturedFrame> MockCameraCapture::capture(
    const std::stop_token stopToken
) {
    if (stopToken.stop_requested() || framesProduced_ >= frameCount_) {
        return std::nullopt;
    }

    auto buffer = std::make_shared<core::Buffer>(
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_)
    );
    const std::uint64_t frameId = firstFrameId_ + framesProduced_;
    for (std::size_t index = 0; index < buffer->size(); ++index) {
        buffer->bytes()[index] =
            static_cast<std::byte>((frameId + index) & 0xFFU);
    }
    ++framesProduced_;

    return core::CapturedFrame(
        frameId,
        core::CapturedFrame::Clock::now(),
        width_,
        height_,
        core::PixelFormat::gray8,
        std::move(buffer)
    );
}

}  // namespace snapsentinel::pipeline
