#include "snapsentinel/pipeline/async_npu_engine.hpp"

#include <dlfcn.h>

#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace snapsentinel::pipeline {
namespace {

using GetProvidersFn = Qnn_ErrorHandle_t (*)(
    const QnnInterface_t*** providerList,
    std::uint32_t* numProviders
);
using LegacyBackendInitializeFn = Qnn_ErrorHandle_t (*)(Qnn_BackendHandle_t* backend);
using LegacyBackendTerminateFn = Qnn_ErrorHandle_t (*)(Qnn_BackendHandle_t backend);

[[noreturn]] void throwQnnError(const char* operation, const Qnn_ErrorHandle_t status) {
    std::fprintf(
        stderr,
        "AsyncNpuEngine: %s failed with QNN status 0x%08x\n",
        operation,
        static_cast<unsigned int>(status)
    );
    throw std::runtime_error(
        std::string(operation) + " failed with QNN status " +
        std::to_string(static_cast<unsigned int>(status))
    );
}

void checkQnnStatus(const char* operation, const Qnn_ErrorHandle_t status) {
    if (status != QNN_SUCCESS) {
        throwQnnError(operation, status);
    }
}

}  // namespace

struct AsyncNpuEngine::State final {
    State(const float threshold, ConfidenceEvaluator evaluator)
        : confidenceThreshold(threshold), confidenceEvaluator(std::move(evaluator)) {}

    void startWorkers(const std::size_t count) {
        workers.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            workers.emplace_back([this] { workerLoop(); });
        }
    }

    void enqueue(std::function<void()> job) {
        {
            std::lock_guard lock(queueMutex);
            jobs.push(std::move(job));
        }
        queueCondition.notify_one();
    }

    void workerLoop() noexcept {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock lock(queueMutex);
                queueCondition.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop();
            }

            try {
                job();
            } catch (const std::exception& error) {
                std::fprintf(stderr, "AsyncNpuEngine: completion handler failed: %s\n", error.what());
            } catch (...) {
                std::fprintf(stderr, "AsyncNpuEngine: completion handler failed\n");
            }
        }
    }

    void finishRequest() noexcept {
        {
            std::lock_guard lock(queueMutex);
            --pendingRequests;
        }
        pendingCondition.notify_all();
    }

    const float confidenceThreshold;
    const ConfidenceEvaluator confidenceEvaluator;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::condition_variable pendingCondition;
    std::queue<std::function<void()>> jobs;
    std::vector<std::thread> workers;
    std::size_t pendingRequests = 0;
    bool stopping = false;

    std::mutex eventMutex;
    std::condition_variable eventCondition;
    bool detectionTriggered = false;
};

struct AsyncNpuEngine::InferenceRequest final {
    std::shared_ptr<State> state;
    Qnn_Tensor_t* outputs = nullptr;
    std::size_t outputCount = 0;
};

AsyncNpuEngine::AsyncNpuEngine(
    std::string contextBinaryPath,
    std::string graphName,
    ConfidenceEvaluator confidenceEvaluator,
    const float confidenceThreshold,
    const std::size_t workerCount
) : state_(std::make_shared<State>(confidenceThreshold, std::move(confidenceEvaluator))) {
    if (graphName.empty()) {
        throw std::invalid_argument("QNN graph name must not be empty");
    }
    if (!(confidenceThreshold >= 0.0F && confidenceThreshold <= 1.0F)) {
        throw std::invalid_argument("Confidence threshold must be between 0 and 1");
    }
    if (!state_->confidenceEvaluator) {
        throw std::invalid_argument("Confidence evaluator must not be empty");
    }
    if (workerCount == 0) {
        throw std::invalid_argument("Worker count must be greater than zero");
    }

    try {
        state_->startWorkers(workerCount);
        initialize(contextBinaryPath, graphName);
    } catch (...) {
        if (context_ != nullptr && qnnInterface_ != nullptr) {
            const Qnn_ErrorHandle_t status =
                qnnInterface_->QNN_INTERFACE_VER_NAME.contextFree(context_, nullptr);
            if (status != QNN_SUCCESS) {
                std::fprintf(stderr, "AsyncNpuEngine: context cleanup failed (0x%08x)\n",
                             static_cast<unsigned int>(status));
            }
            context_ = nullptr;
        }
        if (backend_ != nullptr && qnnInterface_ != nullptr) {
            Qnn_ErrorHandle_t status = QNN_SUCCESS;
            if (legacyBackendInitialization_) {
                dlerror();
                const auto terminate = reinterpret_cast<LegacyBackendTerminateFn>(
                    dlsym(backendLibrary_, "QnnBackend_terminate")
                );
                const char* error = dlerror();
                status = terminate != nullptr && error == nullptr
                    ? terminate(backend_)
                    : qnnInterface_->QNN_INTERFACE_VER_NAME.backendFree(backend_);
            } else {
                status = qnnInterface_->QNN_INTERFACE_VER_NAME.backendFree(backend_);
            }
            if (status != QNN_SUCCESS) {
                std::fprintf(stderr, "AsyncNpuEngine: backend cleanup failed (0x%08x)\n",
                             static_cast<unsigned int>(status));
            }
            backend_ = nullptr;
        }
        {
            std::lock_guard lock(state_->queueMutex);
            state_->stopping = true;
        }
        state_->queueCondition.notify_all();
        for (std::thread& worker : state_->workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        if (backendLibrary_ != nullptr) {
            dlclose(backendLibrary_);
            backendLibrary_ = nullptr;
        }
        throw;
    }
}

AsyncNpuEngine::~AsyncNpuEngine() {
    {
        std::unique_lock lock(state_->queueMutex);
        state_->pendingCondition.wait(lock, [this] { return state_->pendingRequests == 0; });
        state_->stopping = true;
    }
    state_->queueCondition.notify_all();
    for (std::thread& worker : state_->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    if (context_ != nullptr && qnnInterface_ != nullptr) {
        const Qnn_ErrorHandle_t status = qnnInterface_->QNN_INTERFACE_VER_NAME.contextFree(
            context_,
            nullptr
        );
        if (status != QNN_SUCCESS) {
            std::fprintf(stderr, "AsyncNpuEngine: contextFree failed (0x%08x)\n",
                         static_cast<unsigned int>(status));
        }
    }
    if (backend_ != nullptr && qnnInterface_ != nullptr) {
        Qnn_ErrorHandle_t status = QNN_SUCCESS;
        if (legacyBackendInitialization_) {
            dlerror();
            const auto terminate = reinterpret_cast<LegacyBackendTerminateFn>(
                dlsym(backendLibrary_, "QnnBackend_terminate")
            );
            const char* error = dlerror();
            status = terminate != nullptr && error == nullptr
                ? terminate(backend_)
                : qnnInterface_->QNN_INTERFACE_VER_NAME.backendFree(backend_);
        } else {
            status = qnnInterface_->QNN_INTERFACE_VER_NAME.backendFree(backend_);
        }
        if (status != QNN_SUCCESS) {
            std::fprintf(stderr, "AsyncNpuEngine: backend teardown failed (0x%08x)\n",
                         static_cast<unsigned int>(status));
        }
    }
    if (backendLibrary_ != nullptr) {
        dlclose(backendLibrary_);
    }
}

void AsyncNpuEngine::initialize(
    const std::string& contextBinaryPath,
    const std::string& graphName
) {
    backendLibrary_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (backendLibrary_ == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(
            std::string("Unable to load libQnnHtp.so: ") +
            (error != nullptr ? error : "unknown dynamic loader error")
        );
    }

    dlerror();
    const auto getProviders = reinterpret_cast<GetProvidersFn>(
        dlsym(backendLibrary_, "QnnInterface_getProviders")
    );
    const char* symbolError = dlerror();
    if (symbolError != nullptr || getProviders == nullptr) {
        throw std::runtime_error(
            std::string("Unable to resolve QnnInterface_getProviders: ") +
            (symbolError != nullptr ? symbolError : "symbol not found")
        );
    }

    const QnnInterface_t** providers = nullptr;
    std::uint32_t providerCount = 0;
    checkQnnStatus(
        "QnnInterface_getProviders",
        getProviders(&providers, &providerCount)
    );
    for (std::uint32_t index = 0; index < providerCount; ++index) {
        if (providers[index] != nullptr && providers[index]->backendId == 6U) {
            qnnInterface_ = providers[index];
            break;
        }
    }
    if (qnnInterface_ == nullptr) {
        throw std::runtime_error("libQnnHtp.so did not provide a QNN HTP interface");
    }

    dlerror();
    const auto legacyInitialize = reinterpret_cast<LegacyBackendInitializeFn>(
        dlsym(backendLibrary_, "QnnBackend_initialize")
    );
    const char* legacyInitializeError = dlerror();
    if (legacyInitialize != nullptr && legacyInitializeError == nullptr) {
        checkQnnStatus("QnnBackend_initialize", legacyInitialize(&backend_));
        legacyBackendInitialization_ = true;
    } else {
        checkQnnStatus(
            "QnnBackend_create",
            qnnInterface_->QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &backend_)
        );
    }

    std::ifstream binary(contextBinaryPath, std::ios::binary | std::ios::ate);
    if (!binary) {
        throw std::runtime_error("Unable to open QNN context binary: " + contextBinaryPath);
    }
    const std::streampos endPosition = binary.tellg();
    const std::streamoff binarySize = endPosition;
    if (binarySize <= 0 ||
        static_cast<std::uintmax_t>(binarySize) >
            std::numeric_limits<Qnn_ContextBinarySize_t>::max()) {
        throw std::runtime_error("QNN context binary is empty or too large");
    }
    std::vector<char> contextBinary(static_cast<std::size_t>(binarySize));
    binary.seekg(0, std::ios::beg);
    if (!binary.read(contextBinary.data(), static_cast<std::streamsize>(contextBinary.size()))) {
        throw std::runtime_error("Unable to read QNN context binary: " + contextBinaryPath);
    }

    checkQnnStatus(
        "QnnContext_createFromBinary",
        qnnInterface_->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            backend_,
            nullptr,
            nullptr,
            contextBinary.data(),
            static_cast<Qnn_ContextBinarySize_t>(contextBinary.size()),
            &context_,
            nullptr
        )
    );
    checkQnnStatus(
        "QnnGraph_retrieve",
        qnnInterface_->QNN_INTERFACE_VER_NAME.graphRetrieve(context_, graphName.c_str(), &graph_)
    );
}

void AsyncNpuEngine::submit(
    const Qnn_Tensor_t* const inputs,
    const std::size_t inputCount,
    Qnn_Tensor_t* const outputs,
    const std::size_t outputCount
) {
    if (graph_ == nullptr) {
        throw std::logic_error("QNN graph is not initialized");
    }
    if (inputs == nullptr || inputCount == 0 || outputs == nullptr || outputCount == 0) {
        throw std::invalid_argument("Inference requires non-empty input and output tensors");
    }
    if (inputCount > std::numeric_limits<std::uint32_t>::max() ||
        outputCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("QNN tensor count exceeds the API limit");
    }

    auto* request = new InferenceRequest{state_, outputs, outputCount};
    {
        std::lock_guard lock(state_->queueMutex);
        ++state_->pendingRequests;
    }

    const Qnn_ErrorHandle_t status =
        qnnInterface_->QNN_INTERFACE_VER_NAME.graphExecuteAsync(
            graph_,
            inputs,
            static_cast<std::uint32_t>(inputCount),
            outputs,
            static_cast<std::uint32_t>(outputCount),
            nullptr,
            nullptr,
            &AsyncNpuEngine::onQnnExecutionComplete,
            request
        );
    if (status != QNN_SUCCESS) {
        delete request;
        state_->finishRequest();
        throwQnnError("QnnGraph_executeAsync", status);
    }
}

bool AsyncNpuEngine::waitForDetection(const std::chrono::milliseconds timeout) {
    std::unique_lock lock(state_->eventMutex);
    if (!state_->eventCondition.wait_for(
            lock,
            timeout,
            [this] { return state_->detectionTriggered; })) {
        return false;
    }
    state_->detectionTriggered = false;
    return true;
}

bool AsyncNpuEngine::waitUntilIdle(const std::chrono::milliseconds timeout) {
    std::unique_lock lock(state_->queueMutex);
    return state_->pendingCondition.wait_for(
        lock,
        timeout,
        [this] { return state_->pendingRequests == 0; }
    );
}

void AsyncNpuEngine::onQnnExecutionComplete(
    void* const notifyParam,
    const Qnn_NotifyStatus_t notifyStatus
) {
    onInferenceComplete(notifyStatus.error, notifyParam);
}

void AsyncNpuEngine::onInferenceComplete(
    const Qnn_ErrorHandle_t status,
    void* const notifyParam
) noexcept {
    std::unique_ptr<InferenceRequest> request(static_cast<InferenceRequest*>(notifyParam));
    if (request == nullptr) {
        std::fprintf(stderr, "AsyncNpuEngine: completion callback received no request\n");
        return;
    }

    const std::shared_ptr<State> state = request->state;
    try {
        std::shared_ptr<InferenceRequest> sharedRequest(std::move(request));
        state->enqueue([sharedRequest, state, status] {
            try {
                if (status != QNN_SUCCESS) {
                    std::fprintf(
                        stderr,
                        "AsyncNpuEngine: inference completed with QNN status 0x%08x\n",
                        static_cast<unsigned int>(status)
                    );
                } else {
                    const float confidence = state->confidenceEvaluator(
                        sharedRequest->outputs,
                        sharedRequest->outputCount
                    );
                    if (!std::isfinite(confidence) || confidence < 0.0F || confidence > 1.0F) {
                        throw std::invalid_argument(
                            "Confidence evaluator must return a finite value in [0, 1]"
                        );
                    }
                    if (confidence > state->confidenceThreshold) {
                        {
                            std::lock_guard lock(state->eventMutex);
                            state->detectionTriggered = true;
                        }
                        state->eventCondition.notify_one();
                    }
                }
            } catch (const std::exception& error) {
                std::fprintf(
                    stderr,
                    "AsyncNpuEngine: confidence evaluation failed: %s\n",
                    error.what()
                );
            } catch (...) {
                std::fprintf(stderr, "AsyncNpuEngine: confidence evaluation failed\n");
            }
            state->finishRequest();
        });
    } catch (...) {
        state->finishRequest();
        std::fprintf(stderr, "AsyncNpuEngine: unable to queue inference completion\n");
    }
}

}  // namespace snapsentinel::pipeline
