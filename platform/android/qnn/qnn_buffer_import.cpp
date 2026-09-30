#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "qnn_buffer_import.hpp"

#include <QnnMem.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

namespace snapsentinel::platform::android::qnn {
namespace {

bool isQnnSuccess(const Qnn_ErrorHandle_t status) noexcept {
    return status == QNN_SUCCESS || status == QNN_MEMORY_SUCCESS;
}

void logQnnError(const char* operation, const Qnn_ErrorHandle_t status) noexcept {
    std::fprintf(
        stderr,
        "DmaBufferManager: %s failed with QNN status 0x%08x\n",
        operation,
        static_cast<unsigned int>(status)
    );
}

void logCloseError(const int fd) noexcept {
    std::fprintf(
        stderr,
        "DmaBufferManager: close(%d) failed: %s\n",
        fd,
        std::strerror(errno)
    );
}

}  // namespace

DmaMemoryHandle::DmaMemoryHandle(
    const int fd,
    const Qnn_ContextHandle_t context,
    const Qnn_MemHandle_t memoryHandle,
    Qnn_Tensor_t* const tensor,
    const Qnn_TensorMemType_t previousMemoryType,
    const Qnn_MemHandle_t previousMemoryHandle
) noexcept
    : fd_(fd),
      context_(context),
      memoryHandle_(memoryHandle),
      tensor_(tensor),
      previousMemoryType_(previousMemoryType),
      previousMemoryHandle_(previousMemoryHandle) {}

DmaMemoryHandle::~DmaMemoryHandle() {
    release();
}

DmaMemoryHandle::DmaMemoryHandle(DmaMemoryHandle&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      context_(std::exchange(other.context_, nullptr)),
      memoryHandle_(std::exchange(other.memoryHandle_, nullptr)),
      tensor_(std::exchange(other.tensor_, nullptr)),
      previousMemoryType_(other.previousMemoryType_),
      previousMemoryHandle_(std::exchange(other.previousMemoryHandle_, nullptr)) {}

DmaMemoryHandle& DmaMemoryHandle::operator=(DmaMemoryHandle&& other) noexcept {
    if (this != &other) {
        release();
        fd_ = std::exchange(other.fd_, -1);
        context_ = std::exchange(other.context_, nullptr);
        memoryHandle_ = std::exchange(other.memoryHandle_, nullptr);
        tensor_ = std::exchange(other.tensor_, nullptr);
        previousMemoryType_ = other.previousMemoryType_;
        previousMemoryHandle_ = std::exchange(other.previousMemoryHandle_, nullptr);
    }
    return *this;
}

int DmaMemoryHandle::fileDescriptor() const noexcept {
    return fd_;
}

Qnn_MemHandle_t DmaMemoryHandle::qnnMemoryHandle() const noexcept {
    return memoryHandle_;
}

DmaMemoryHandle::operator bool() const noexcept {
    return fd_ >= 0 && memoryHandle_ != nullptr;
}

void DmaMemoryHandle::release() noexcept {
    if (tensor_ != nullptr &&
        tensor_->version == QNN_TENSOR_VERSION_1 &&
        tensor_->v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE &&
        tensor_->v1.memHandle == memoryHandle_) {
        tensor_->v1.memType = previousMemoryType_;
        tensor_->v1.memHandle = previousMemoryHandle_;
    }

    if (context_ != nullptr && memoryHandle_ != nullptr) {
        const Qnn_ErrorHandle_t status =
            QnnMemory_deRegister(context_, &memoryHandle_, 1);
        if (!isQnnSuccess(status)) {
            logQnnError("QnnMemory_deRegister", status);
        }
    }

    if (fd_ >= 0 && close(fd_) != 0) {
        logCloseError(fd_);
    }

    fd_ = -1;
    context_ = nullptr;
    memoryHandle_ = nullptr;
    tensor_ = nullptr;
    previousMemoryHandle_ = nullptr;
}

int DmaBufferManager::allocateDmaBuffer(const std::size_t size_bytes) {
#if defined(__linux__) && defined(SYS_memfd_create)
    if (size_bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
        throw std::length_error("DMA buffer size exceeds the supported file size");
    }

    const int fd = static_cast<int>(syscall(SYS_memfd_create, "snapsentinel-dma", MFD_CLOEXEC));
    if (fd < 0) {
        const int error = errno;
        std::fprintf(
            stderr,
            "DmaBufferManager: memfd_create failed: %s\n",
            std::strerror(error)
        );
        throw std::system_error(error, std::generic_category(), "memfd_create");
    }

    if (ftruncate(fd, static_cast<off_t>(size_bytes)) != 0) {
        const int error = errno;
        if (close(fd) != 0) {
            logCloseError(fd);
        }
        std::fprintf(
            stderr,
            "DmaBufferManager: ftruncate failed: %s\n",
            std::strerror(error)
        );
        throw std::system_error(error, std::generic_category(), "ftruncate DMA buffer");
    }

    return fd;
#else
    (void)size_bytes;
    throw std::runtime_error("DMA buffer allocation requires Linux memfd_create support");
#endif
}

DmaMemoryHandle DmaBufferManager::registerWithQnn(
    const int fd,
    const Qnn_ContextHandle_t context,
    Qnn_Tensor_t* const tensor
) {
    if (fd < 0) {
        throw std::invalid_argument("registerWithQnn requires a valid file descriptor");
    }
    if (context == nullptr) {
        throw std::invalid_argument("registerWithQnn requires a valid QNN context");
    }
    if (tensor == nullptr) {
        throw std::invalid_argument("registerWithQnn requires a valid tensor");
    }
    if (tensor->version != QNN_TENSOR_VERSION_1) {
        throw std::invalid_argument("registerWithQnn currently supports QNN tensor version 1");
    }

    struct stat bufferStatus {};
    if (fstat(fd, &bufferStatus) != 0) {
        const int error = errno;
        std::fprintf(
            stderr,
            "DmaBufferManager: fstat(%d) failed: %s\n",
            fd,
            std::strerror(error)
        );
        throw std::system_error(error, std::generic_category(), "fstat DMA buffer");
    }
    if (bufferStatus.st_size <= 0 ||
        static_cast<unsigned long long>(bufferStatus.st_size) >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("QNN DMA buffer size must be between 1 and UINT32_MAX bytes");
    }

    Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
    descriptor.memType = QNN_MEM_TYPE_ION;
    descriptor.memInfo.ionInfo.fd = fd;
    descriptor.memInfo.ionInfo.offset = 0;
    descriptor.memInfo.ionInfo.size = static_cast<std::uint32_t>(bufferStatus.st_size);

    Qnn_MemHandle_t memoryHandle = nullptr;
    const Qnn_ErrorHandle_t status = QnnMemory_register(context, &descriptor, 1, &memoryHandle);
    if (!isQnnSuccess(status)) {
        logQnnError("QnnMemory_register", status);
        throw std::runtime_error(
            "QnnMemory_register failed with status " +
            std::to_string(static_cast<unsigned int>(status))
        );
    }
    if (memoryHandle == nullptr) {
        throw std::runtime_error("QnnMemory_register succeeded without returning a memory handle");
    }

    const Qnn_TensorMemType_t previousMemoryType = tensor->v1.memType;
    const Qnn_MemHandle_t previousMemoryHandle = tensor->v1.memHandle;
    tensor->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor->v1.memHandle = memoryHandle;

    return DmaMemoryHandle(
        fd,
        context,
        memoryHandle,
        tensor,
        previousMemoryType,
        previousMemoryHandle
    );
}

}  // namespace snapsentinel::platform::android::qnn
