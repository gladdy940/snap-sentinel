#pragma once

#include "snapsentinel/core/frame.hpp"
#include "snapsentinel/inference/tensor.hpp"

#include <memory>

namespace snapsentinel::inference {

class TensorPreparer {
public:
    virtual ~TensorPreparer() = default;

    // The returned request must own or lease all storage needed after this call.
    [[nodiscard]] virtual std::unique_ptr<PreparedInference> prepare(
        const core::CapturedFrame& frame
    ) = 0;
};

class DeterministicTensorPreparer final : public TensorPreparer {
public:
    [[nodiscard]] std::unique_ptr<PreparedInference> prepare(
        const core::CapturedFrame& frame
    ) override;
};

}  // namespace snapsentinel::inference
