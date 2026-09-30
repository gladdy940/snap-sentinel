#pragma once

#include "snapsentinel/core/frame.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>

namespace snapsentinel::pipeline {

using CapturedFrame = core::CapturedFrame;

class ICameraCapture {
public:
    virtual ~ICameraCapture() = default;

    [[nodiscard]] virtual std::optional<core::CapturedFrame> capture(
        std::stop_token stopToken
    ) = 0;
};

class MockCameraCapture final : public ICameraCapture {
public:
    MockCameraCapture(
        std::uint32_t width,
        std::uint32_t height,
        std::size_t frameCount,
        std::uint64_t firstFrameId = 0
    );

    [[nodiscard]] std::optional<core::CapturedFrame> capture(
        std::stop_token stopToken
    ) override;

private:
    std::uint32_t width_;
    std::uint32_t height_;
    std::size_t frameCount_;
    std::uint64_t firstFrameId_;
    std::size_t framesProduced_ = 0;
};

}  // namespace snapsentinel::pipeline
