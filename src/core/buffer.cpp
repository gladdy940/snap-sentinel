#include "snapsentinel/core/buffer.hpp"

#include <utility>

namespace snapsentinel::core {

Buffer::Buffer(const std::size_t size) : storage_(size) {}

Buffer::Buffer(std::vector<std::byte> storage) noexcept
    : storage_(std::move(storage)) {}

std::byte* Buffer::data() noexcept {
    return storage_.data();
}

const std::byte* Buffer::data() const noexcept {
    return storage_.data();
}

std::size_t Buffer::size() const noexcept {
    return storage_.size();
}

bool Buffer::empty() const noexcept {
    return storage_.empty();
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
