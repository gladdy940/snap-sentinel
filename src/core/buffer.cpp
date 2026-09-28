#include "snapsentinel/core/buffer.hpp"

namespace snapsentinel::core {

Buffer::Buffer(const std::size_t size) : storage_(size) {}

std::size_t Buffer::size() const noexcept {
    return storage_.size();
}

std::span<std::byte> Buffer::bytes() noexcept {
    return storage_;
}

std::span<const std::byte> Buffer::bytes() const noexcept {
    return storage_;
}

Buffer BufferAllocator::allocate(const std::size_t size) const {
    return Buffer(size);
}

}  // namespace snapsentinel::core
