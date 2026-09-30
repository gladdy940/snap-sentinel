#include "snapsentinel/inference/inference_engine.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace snapsentinel::inference {

void MockInferenceEngine::submit(
    const PreparedInference& request,
    CompletionCallback completion
) {
    if (!completion) {
        throw std::invalid_argument("Inference completion callback must not be empty");
    }
    const auto start = std::chrono::steady_clock::now();
    const auto* const batch = dynamic_cast<const TensorBatch*>(&request);
    if (batch == nullptr || batch->inputs.empty()) {
        throw std::invalid_argument("Mock inference requires a tensor batch with an input");
    }
    const Tensor& input = batch->inputs.front();
    if (input.elementType != TensorElementType::float32) {
        throw std::invalid_argument("Mock inference requires a float32 input tensor");
    }
    const std::size_t elementCount = input.elementCount();
    if (elementCount == 0 || elementCount > input.storage.size() / sizeof(float) ||
        input.storage.size() != elementCount * sizeof(float)) {
        throw std::invalid_argument("Mock inference received inconsistent tensor storage");
    }

    double total = 0.0;
    for (std::size_t index = 0; index < elementCount; ++index) {
        float value = 0.0F;
        std::memcpy(&value, input.storage.data() + index * sizeof(float), sizeof(float));
        total += value;
    }

    InferenceResult result;
    result.score = total / static_cast<double>(elementCount);
    result.inferenceTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start
    );
    completion(std::move(result));
}

}  // namespace snapsentinel::inference
