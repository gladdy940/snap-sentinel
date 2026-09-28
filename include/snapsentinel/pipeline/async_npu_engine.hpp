#pragma once

#include <QnnInterface.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace snapsentinel::pipeline {

class AsyncNpuEngine final {
public:
    // Evaluated on a worker thread after successful inference; return the model's
    // detection confidence in the [0, 1] range.
    using ConfidenceEvaluator =
        std::function<float(const Qnn_Tensor_t* outputs, std::size_t outputCount)>;

    AsyncNpuEngine(
        std::string contextBinaryPath,
        std::string graphName,
        ConfidenceEvaluator confidenceEvaluator,
        float confidenceThreshold = 0.75F,
        std::size_t workerCount = 2
    );
    ~AsyncNpuEngine();

    AsyncNpuEngine(const AsyncNpuEngine&) = delete;
    AsyncNpuEngine& operator=(const AsyncNpuEngine&) = delete;
    AsyncNpuEngine(AsyncNpuEngine&&) = delete;
    AsyncNpuEngine& operator=(AsyncNpuEngine&&) = delete;

    // Tensor arrays and their backing buffers must remain valid until waitUntilIdle succeeds.
    void submit(
        const Qnn_Tensor_t* inputs,
        std::size_t inputCount,
        Qnn_Tensor_t* outputs,
        std::size_t outputCount
    );

    [[nodiscard]] bool waitForDetection(std::chrono::milliseconds timeout);
    [[nodiscard]] bool waitUntilIdle(std::chrono::milliseconds timeout);

private:
    struct State;
    struct InferenceRequest;

    static void onInferenceComplete(Qnn_ErrorHandle_t status, void* notifyParam) noexcept;
    static void onQnnExecutionComplete(void* notifyParam, Qnn_NotifyStatus_t notifyStatus);

    void initialize(const std::string& contextBinaryPath, const std::string& graphName);

    std::shared_ptr<State> state_;
    void* backendLibrary_ = nullptr;
    const QnnInterface_t* qnnInterface_ = nullptr;
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_ContextHandle_t context_ = nullptr;
    Qnn_GraphHandle_t graph_ = nullptr;
    bool legacyBackendInitialization_ = false;
};

}  // namespace snapsentinel::pipeline
