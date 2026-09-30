#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace snapsentinel::core {

class Buffer final {
public:
    Buffer() = default;
    explicit Buffer(std::size_t size);
    explicit Buffer(std::vector<std::byte> storage) noexcept;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    [[nodiscard]] std::byte* data() noexcept;
    [[nodiscard]] const std::byte* data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

private:
    std::vector<std::byte> storage_;
};

class BufferAllocator final {
public:
    [[nodiscard]] Buffer allocate(std::size_t size) const;
};

}  // namespace snapsentinel::core
