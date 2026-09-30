#include "snapsentinel/inference/inference_engine.hpp"
#include "snapsentinel/inference/tensor.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

namespace snapsentinel::inference {
namespace {

TEST(MockInferenceEngineTest, ImplementsGenericInterfaceAndCompletesPreparedTensor) {
    TensorBatch batch;
    Tensor input;
    input.shape = {2};
    input.elementType = TensorElementType::float32;
    input.storage = core::Buffer(2 * sizeof(float));
    const float values[] = {0.25F, 0.75F};
    std::memcpy(input.storage.data(), values, sizeof(values));
    batch.inputs.push_back(std::move(input));

    std::unique_ptr<InferenceEngine> engine = std::make_unique<MockInferenceEngine>();
    bool completed = false;
    InferenceResult result;
    engine->submit(batch, [&completed, &result](InferenceResult inferenceResult) {
        result = std::move(inferenceResult);
        completed = true;
    });

    ASSERT_TRUE(completed);
    EXPECT_TRUE(result.succeeded());
    EXPECT_DOUBLE_EQ(result.score, 0.5);
    EXPECT_GE(result.inferenceTime.count(), 0);
}

TEST(TensorTest, StoresPortableTensorMetadataAndBytes) {
    TensorBatch batch;
    Tensor tensor;
    tensor.shape = {1, 2, 2, 1};
    tensor.elementType = TensorElementType::uint8;
    tensor.layout = TensorLayout::nhwc;
    tensor.storage = core::Buffer(4);
    tensor.storage.bytes()[0] = std::byte{0x2A};
    batch.inputs.push_back(std::move(tensor));

    ASSERT_EQ(batch.inputs.size(), 1U);
    EXPECT_EQ(batch.inputs.front().shape.size(), 4U);
    EXPECT_EQ(batch.inputs.front().storage.size(), 4U);
    EXPECT_EQ(batch.inputs.front().storage.bytes()[0], std::byte{0x2A});
}

}  // namespace
}  // namespace snapsentinel::inference
