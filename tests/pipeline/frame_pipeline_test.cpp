#include "snapsentinel/pipeline/frame_pipeline.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <utility>
#include <vector>

namespace snapsentinel::pipeline {
namespace {

class TestTensorPreparer final : public inference::TensorPreparer {
public:
    [[nodiscard]] std::unique_ptr<inference::PreparedInference> prepare(
        const core::CapturedFrame& frame
    ) override {
        EXPECT_EQ(frame.width, 2U);
        EXPECT_EQ(frame.height, 2U);
        return std::make_unique<inference::TensorBatch>();
    }
};

class ImmediateInferenceEngine final : public inference::InferenceEngine {
public:
    void submit(
        const inference::PreparedInference& request,
        CompletionCallback completion
    ) override {
        EXPECT_NE(dynamic_cast<const inference::TensorBatch*>(&request), nullptr);
        completion(inference::InferenceResult{});
    }
};

TEST(FramePipelineTest, PreparesFramesAndReportsInferenceCompletion) {
    struct Completion final {
        std::uint64_t frameId;
        bool succeeded;
    };

    TestTensorPreparer tensorPreparer;
    ImmediateInferenceEngine inferenceEngine;
    std::promise<Completion> completionPromise;
    std::future<Completion> completion = completionPromise.get_future();
    FramePipeline pipeline(
        tensorPreparer,
        inferenceEngine,
        [&completionPromise](const std::uint64_t frameId, inference::InferenceResult result) {
            completionPromise.set_value(Completion{frameId, result.succeeded()});
        }
    );

    pipeline.submit(42, core::CapturedFrame{2, 2, std::vector<std::byte>(4)});
    const Completion result = completion.get();
    EXPECT_EQ(result.frameId, 42U);
    EXPECT_TRUE(result.succeeded);
}

}  // namespace
}  // namespace snapsentinel::pipeline
