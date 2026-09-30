#include "snapsentinel/core/frame.hpp"

#include <utility>

namespace snapsentinel::core {

CapturedFrame::CapturedFrame(
    const std::uint32_t frameWidth,
    const std::uint32_t frameHeight,
    std::vector<std::byte> framePixels
)
    : width(frameWidth),
      height(frameHeight),
      buffer(std::make_shared<Buffer>(std::move(framePixels))) {}

CapturedFrame::CapturedFrame(
    const std::uint64_t frameId,
    const Clock::time_point captureTime,
    const std::uint32_t frameWidth,
    const std::uint32_t frameHeight,
    const PixelFormat format,
    std::shared_ptr<Buffer> frameBuffer
)
    : id(frameId),
      timestamp(captureTime),
      width(frameWidth),
      height(frameHeight),
      pixelFormat(format),
      buffer(std::move(frameBuffer)) {}

std::span<const std::byte> CapturedFrame::pixels() const noexcept {
    return buffer != nullptr
        ? std::span<const std::byte>(buffer->bytes())
        : std::span<const std::byte>{};
}

}  // namespace snapsentinel::core
