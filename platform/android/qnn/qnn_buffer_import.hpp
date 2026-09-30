#pragma once

#include <QnnMem.h>
#include <QnnTypes.h>

#include <cstddef>

namespace snapsentinel::platform::android::qnn {

class DmaMemoryHandle final {
public:
    DmaMemoryHandle() noexcept = default;
    ~DmaMemoryHandle();

    DmaMemoryHandle(const DmaMemoryHandle&) = delete;
    DmaMemoryHandle& operator=(const DmaMemoryHandle&) = delete;
    DmaMemoryHandle(DmaMemoryHandle&& other) noexcept;
    DmaMemoryHandle& operator=(DmaMemoryHandle&& other) noexcept;

    [[nodiscard]] int fileDescriptor() const noexcept;
    [[nodiscard]] Qnn_MemHandle_t qnnMemoryHandle() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    friend class DmaBufferManager;

    DmaMemoryHandle(
        int fd,
        Qnn_ContextHandle_t context,
        Qnn_MemHandle_t memoryHandle,
        Qnn_Tensor_t* tensor,
        Qnn_TensorMemType_t previousMemoryType,
        Qnn_MemHandle_t previousMemoryHandle
    ) noexcept;

    void release() noexcept;

    int fd_ = -1;
    Qnn_ContextHandle_t context_ = nullptr;
    Qnn_MemHandle_t memoryHandle_ = nullptr;
    Qnn_Tensor_t* tensor_ = nullptr;
    Qnn_TensorMemType_t previousMemoryType_{};
    Qnn_MemHandle_t previousMemoryHandle_ = nullptr;
};

class DmaBufferManager final {
public:
    [[nodiscard]] static int allocateDmaBuffer(std::size_t size_bytes);

    // Ownership of fd transfers to the returned handle only on success.
    // The tensor must remain alive until the returned handle is destroyed.
    [[nodiscard]] static DmaMemoryHandle registerWithQnn(
        int fd,
        Qnn_ContextHandle_t context,
        Qnn_Tensor_t* tensor
    );
};

}  // namespace snapsentinel::platform::android::qnn
