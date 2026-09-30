#include "snapsentinel/inference/tensor_preparer.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace snapsentinel::inference {

std::unique_ptr<PreparedInference> DeterministicTensorPreparer::prepare(
    const core::CapturedFrame& frame
) {
    if (frame.pixelFormat != core::PixelFormat::gray8) {
        throw std::invalid_argument("Tensor preparer requires gray8 frames");
    }
    if (frame.buffer == nullptr) {
        throw std::invalid_argument("Captured frame has no pixel buffer");
    }
    const std::size_t width = frame.width;
    const std::size_t height = frame.height;
    if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("Captured frame dimensions overflow size_t");
    }
    const std::size_t pixelCount = width * height;
    const std::span<const std::byte> pixels = frame.pixels();
    if (pixels.size() != pixelCount) {
        throw std::invalid_argument("Captured frame buffer size does not match its dimensions");
    }
    if (pixelCount > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("Prepared tensor size overflows size_t");
    }

    Tensor input;
    input.shape = {1, height, width, 1};
    input.elementType = TensorElementType::float32;
    input.layout = TensorLayout::nhwc;
    input.storage = core::Buffer(pixelCount * sizeof(float));
    for (std::size_t index = 0; index < pixelCount; ++index) {
        const float normalized =
            static_cast<float>(std::to_integer<unsigned int>(pixels[index])) / 255.0F;
        std::memcpy(input.storage.data() + index * sizeof(float), &normalized, sizeof(float));
    }

    auto batch = std::make_unique<TensorBatch>();
    batch->inputs.push_back(std::move(input));
    return batch;
}

}  // namespace snapsentinel::inference
