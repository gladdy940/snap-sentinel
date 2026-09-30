#include "snapsentinel/benchmark/benchmark_runner.hpp"
#include "snapsentinel/inference/inference_engine.hpp"
#include "snapsentinel/inference/tensor_preparer.hpp"
#include "snapsentinel/pipeline/async_frame_loop.hpp"
#include "snapsentinel/pipeline/camera_capture.hpp"
#include "snapsentinel/pipeline/frame_pipeline.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <vector>

namespace snapsentinel::pipeline {
namespace {

TEST(MockCameraCaptureTest, ProducesIncreasingIdsAndDeterministicPixels) {
    MockCameraCapture camera(2, 2, 2, 17);

    const auto first = camera.capture(std::stop_token{});
    const auto second = camera.capture(std::stop_token{});

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(first->id, 17U);
    EXPECT_EQ(second->id, 18U);
    ASSERT_EQ(first->pixels().size(), 4U);
    EXPECT_EQ(first->pixels()[0], std::byte{17});
    EXPECT_EQ(first->pixels()[3], std::byte{20});
    EXPECT_TRUE(first->timestamp <= second->timestamp);
    EXPECT_FALSE(camera.capture(std::stop_token{}).has_value());
}

TEST(AsyncFrameLoopTest, DrainsBoundedQueueWhenStopped) {
    MockCameraCapture camera(2, 2, 40);
    std::mutex handledMutex;
    std::condition_variable handledCondition;
    std::size_t handledFrames = 0;
    AsyncFrameLoop loop(
        camera,
        [&handledFrames, &handledMutex, &handledCondition](core::CapturedFrame) {
            {
                std::lock_guard lock(handledMutex);
                ++handledFrames;
            }
            handledCondition.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        },
        1
    );

    loop.start();
    {
        std::unique_lock lock(handledMutex);
        ASSERT_TRUE(handledCondition.wait_for(
            lock,
            std::chrono::seconds(2),
            [&handledFrames] { return handledFrames > 0; }
        ));
    }
    loop.stop();

    EXPECT_GT(handledFrames, 0U);
    EXPECT_LE(handledFrames, 40U);
}

TEST(AsyncFrameLoopTest, PropagatesHandlerFailuresFromStop) {
    MockCameraCapture camera(2, 2, 4);
    std::promise<void> handlerEnteredPromise;
    std::future<void> handlerEntered = handlerEnteredPromise.get_future();
    AsyncFrameLoop loop(
        camera,
        [&handlerEnteredPromise](core::CapturedFrame) {
            handlerEnteredPromise.set_value();
            throw std::runtime_error("handler failed");
        },
        1
    );

    loop.start();
    ASSERT_EQ(handlerEntered.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_THROW(loop.stop(), std::runtime_error);
}

TEST(FramePipelineTest, RunsCameraThroughTensorPreparationAndMockInference) {
    constexpr std::size_t frameCount = 8;
    MockCameraCapture camera(4, 4, frameCount, 200);
    inference::DeterministicTensorPreparer tensorPreparer;
    inference::MockInferenceEngine inferenceEngine;
    std::mutex completionMutex;
    std::condition_variable completionCondition;
    std::vector<std::uint64_t> frameIds;
    std::vector<double> scores;
    FramePipeline pipeline(
        camera,
        tensorPreparer,
        inferenceEngine,
        [&](const std::uint64_t frameId, inference::InferenceResult result) {
            std::lock_guard lock(completionMutex);
            frameIds.push_back(frameId);
            EXPECT_TRUE(result.succeeded());
            EXPECT_GE(result.pipelineLatency.count(), 0);
            EXPECT_GE(result.inferenceTime.count(), 0);
            scores.push_back(result.score);
            completionCondition.notify_all();
        },
        2
    );

    pipeline.start();
    {
        std::unique_lock lock(completionMutex);
        ASSERT_TRUE(completionCondition.wait_for(
            lock,
            std::chrono::seconds(5),
            [&frameIds] { return frameIds.size() == frameCount; }
        ));
    }
    pipeline.stop();

    ASSERT_EQ(frameIds.size(), frameCount);
    ASSERT_EQ(scores.size(), frameCount);
    for (std::size_t index = 0; index < frameCount; ++index) {
        EXPECT_EQ(frameIds[index], 200U + index);
        EXPECT_GE(scores[index], 0.0);
        EXPECT_LE(scores[index], 1.0);
    }
    const PipelineStatistics stats = pipeline.statistics();
    EXPECT_EQ(stats.completedFrames, frameCount);
    EXPECT_EQ(stats.successfulFrames, frameCount);
    EXPECT_EQ(stats.failedFrames, 0U);
}

TEST(BenchmarkRunnerTest, MeasuresPipelineAndWritesPerFrameCsv) {
    constexpr std::size_t frameCount = 5;
    const std::filesystem::path csvPath =
        std::filesystem::temp_directory_path() / "snapsentinel_pipeline_test.csv";
    std::error_code error;
    std::filesystem::remove(csvPath, error);

    MockCameraCapture camera(4, 4, frameCount, 300);
    inference::DeterministicTensorPreparer tensorPreparer;
    inference::MockInferenceEngine inferenceEngine;
    const BenchmarkSummary summary = BenchmarkRunner::runPipeline(
        camera,
        tensorPreparer,
        inferenceEngine,
        frameCount,
        csvPath
    );

    EXPECT_EQ(summary.completedFrames, frameCount);
    EXPECT_EQ(summary.failedFrames, 0U);
    EXPECT_GT(summary.elapsedSeconds, 0.0);
    EXPECT_GT(summary.framesPerSecond, 0.0);
    EXPECT_GE(summary.minLatencyMs, 0.0);
    EXPECT_GE(summary.maxLatencyMs, summary.minLatencyMs);
    EXPECT_GE(summary.meanLatencyMs, summary.minLatencyMs);
    EXPECT_LE(summary.meanLatencyMs, summary.maxLatencyMs);
    EXPECT_GE(summary.meanInferenceTimeMs, 0.0);

    std::ifstream csv(csvPath);
    ASSERT_TRUE(csv.good());
    std::size_t lineCount = 0;
    for (std::string line; std::getline(csv, line);) {
        ++lineCount;
    }
    EXPECT_EQ(lineCount, frameCount + 1);
    std::filesystem::remove(csvPath, error);
}

}  // namespace
}  // namespace snapsentinel::pipeline
