#pragma once

#include "snapsentinel/inference/tensor.hpp"

#include <chrono>
#include <exception>
#include <functional>

namespace snapsentinel::inference {

struct InferenceResult final {
    std::exception_ptr failure;
    std::chrono::nanoseconds inferenceTime{0};
    std::chrono::nanoseconds pipelineLatency{0};
    double score = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return failure == nullptr;
    }
};

class InferenceEngine {
public:
    using CompletionCallback = std::function<void(InferenceResult)>;

    virtual ~InferenceEngine() = default;

    // The prepared request must remain alive until completion is invoked.
    virtual void submit(
        const PreparedInference& request,
        CompletionCallback completion
    ) = 0;
};

class MockInferenceEngine final : public InferenceEngine {
public:
    void submit(
        const PreparedInference& request,
        CompletionCallback completion
    ) override;
};

}  // namespace snapsentinel::inference
