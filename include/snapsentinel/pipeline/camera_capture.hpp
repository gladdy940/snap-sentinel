#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <vector>

namespace snapsentinel::pipeline {

struct CapturedFrame {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::byte> pixels;
};

class ICameraCapture {
public:
    virtual ~ICameraCapture() = default;

    [[nodiscard]] virtual std::optional<CapturedFrame> capture(std::stop_token stopToken) = 0;
};

}  // namespace snapsentinel::pipeline
