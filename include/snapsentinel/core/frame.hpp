#pragma once

#include "snapsentinel/core/buffer.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace snapsentinel::core {

enum class PixelFormat {
    gray8
};

struct CapturedFrame final {
    using Clock = std::chrono::steady_clock;

    CapturedFrame() = default;
    CapturedFrame(
        std::uint32_t frameWidth,
        std::uint32_t frameHeight,
        std::vector<std::byte> framePixels
    );
    CapturedFrame(
        std::uint64_t frameId,
        Clock::time_point captureTime,
        std::uint32_t frameWidth,
        std::uint32_t frameHeight,
        PixelFormat format,
        std::shared_ptr<Buffer> frameBuffer
    );

    [[nodiscard]] std::span<const std::byte> pixels() const noexcept;

    std::uint64_t id = 0;
    Clock::time_point timestamp = Clock::now();
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    PixelFormat pixelFormat = PixelFormat::gray8;
    std::shared_ptr<Buffer> buffer;
};

}  // namespace snapsentinel::core
