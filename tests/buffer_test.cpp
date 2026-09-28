#include "snapsentinel/core/buffer.hpp"

#include <gtest/gtest.h>

#include <cstddef>

namespace snapsentinel::core {
namespace {

TEST(BufferAllocatorTest, AllocatesRequestedNumberOfBytes) {
    const Buffer buffer = BufferAllocator{}.allocate(128);

    EXPECT_EQ(buffer.size(), 128U);
    EXPECT_EQ(buffer.bytes().size(), 128U);
}

TEST(BufferAllocatorTest, AllocatesAnEmptyBuffer) {
    const Buffer buffer = BufferAllocator{}.allocate(0);

    EXPECT_TRUE(buffer.bytes().empty());
}

TEST(BufferAllocatorTest, ProvidesMutableStorage) {
    Buffer buffer = BufferAllocator{}.allocate(1);
    buffer.bytes()[0] = std::byte{0x2A};

    EXPECT_EQ(buffer.bytes()[0], std::byte{0x2A});
}

}  // namespace
}  // namespace snapsentinel::core
