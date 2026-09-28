#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace snapsentinel::core {

class Buffer final {
public:
    explicit Buffer(std::size_t size);

    [[nodiscard]] std::size_t size() const noexcept;
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
