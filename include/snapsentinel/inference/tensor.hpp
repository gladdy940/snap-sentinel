#pragma once

#include "snapsentinel/core/buffer.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace snapsentinel::inference {

enum class TensorElementType {
    uint8,
    int8,
    int16,
    float32
};

enum class TensorLayout {
    unspecified,
    nhwc,
    nchw
};

struct Tensor final {
    std::vector<std::size_t> shape;
    TensorElementType elementType = TensorElementType::uint8;
    TensorLayout layout = TensorLayout::unspecified;
    core::Buffer storage{0};

    [[nodiscard]] std::size_t elementCount() const {
        std::size_t count = 1;
        for (const std::size_t dimension : shape) {
            if (dimension != 0 &&
                count > std::numeric_limits<std::size_t>::max() / dimension) {
                throw std::overflow_error("Tensor element count overflows size_t");
            }
            count *= dimension;
        }
        return count;
    }
};

class PreparedInference {
public:
    virtual ~PreparedInference() = default;
};

class TensorBatch final : public PreparedInference {
public:
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
};

}  // namespace snapsentinel::inference
