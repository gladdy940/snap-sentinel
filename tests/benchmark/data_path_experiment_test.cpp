#include "snapsentinel/benchmark/data_path_experiment.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace snapsentinel::benchmark {
namespace {

TEST(DataPathInstrumentationTest, CountsNamedStagingCopiesAndAllocationCategories) {
    DataPathInstrumentation instrumentation;
    instrumentation.recordFullFrameStagingCopy(4);
    instrumentation.recordFullTensorStagingCopy(16);
    instrumentation.recordCapturedFramePayloadAllocation();
    instrumentation.recordBaselineScratchStagingAllocation();
    instrumentation.recordTensorPayloadAllocation();
    instrumentation.recordFrameProcessed();
    instrumentation.recordPreprocessingTime(123);

    const DataPathCounters recorded = instrumentation.snapshot();
    EXPECT_EQ(recorded.fullFrameStagingCopies, 1U);
    EXPECT_EQ(recorded.fullTensorStagingCopies, 1U);
    EXPECT_EQ(recorded.instrumentedStagingBytesCopied, 20U);
    EXPECT_EQ(recorded.capturedFramePayloadAllocations, 1U);
    EXPECT_EQ(recorded.baselineScratchStagingAllocations, 1U);
    EXPECT_EQ(recorded.tensorPayloadAllocations, 1U);
    EXPECT_EQ(recorded.framesProcessed, 1U);
    EXPECT_EQ(recorded.preprocessingNanoseconds, 123U);

    instrumentation.reset();
    const DataPathCounters reset = instrumentation.snapshot();
    EXPECT_EQ(reset.fullFrameStagingCopies, 0U);
    EXPECT_EQ(reset.fullTensorStagingCopies, 0U);
    EXPECT_EQ(reset.instrumentedStagingBytesCopied, 0U);
    EXPECT_EQ(reset.capturedFramePayloadAllocations, 0U);
    EXPECT_EQ(reset.baselineScratchStagingAllocations, 0U);
    EXPECT_EQ(reset.tensorPayloadAllocations, 0U);
    EXPECT_EQ(reset.framesProcessed, 0U);
    EXPECT_EQ(reset.preprocessingNanoseconds, 0U);
}

TEST(DataPathExperimentTest, ReportsExpectedApplicationCopiesAndEquivalentResults) {
    const std::filesystem::path csvPath =
        std::filesystem::temp_directory_path() / "snapsentinel_data_path_test.csv";
    std::error_code error;
    std::filesystem::remove(csvPath, error);

    const DataPathExperimentResult result = runDataPathExperiment(DataPathExperimentOptions{
        2,
        2,
        2,
        2,
        1,
        csvPath
    });

    EXPECT_EQ(result.baselineAccounting.framesProcessed, 4U);
    EXPECT_EQ(result.snapSentinelAccounting.framesProcessed, 4U);

    EXPECT_EQ(result.baselineAccounting.fullFrameStagingCopies, 4U);
    EXPECT_EQ(result.baselineAccounting.fullTensorStagingCopies, 4U);
    EXPECT_EQ(result.baselineAccounting.instrumentedStagingBytesCopied, 80U);
    EXPECT_EQ(result.baselineAccounting.capturedFramePayloadAllocations, 4U);
    EXPECT_EQ(result.baselineAccounting.baselineScratchStagingAllocations, 8U);
    EXPECT_EQ(result.baselineAccounting.tensorPayloadAllocations, 4U);

    EXPECT_EQ(result.snapSentinelAccounting.fullFrameStagingCopies, 0U);
    EXPECT_EQ(result.snapSentinelAccounting.fullTensorStagingCopies, 0U);
    EXPECT_EQ(result.snapSentinelAccounting.instrumentedStagingBytesCopied, 0U);
    EXPECT_EQ(result.snapSentinelAccounting.capturedFramePayloadAllocations, 4U);
    EXPECT_EQ(result.snapSentinelAccounting.baselineScratchStagingAllocations, 0U);
    EXPECT_EQ(result.snapSentinelAccounting.tensorPayloadAllocations, 4U);

    std::ifstream csv(csvPath);
    ASSERT_TRUE(csv);
    const std::string contents{
        std::istreambuf_iterator<char>{csv},
        std::istreambuf_iterator<char>{}
    };
    EXPECT_NE(contents.find(
                  "measurement_mode,build_configuration,pipeline,iteration,frames,width,"
                  "height,warmup_frames,average_latency_ms"
              ),
              std::string::npos);
    EXPECT_NE(contents.find("accounting,"), std::string::npos);
    EXPECT_NE(contents.find("performance,"), std::string::npos);
    EXPECT_NE(contents.find("full_frame_staging_copies"), std::string::npos);
    EXPECT_NE(contents.find("captured_frame_allocations"), std::string::npos);
    std::istringstream csvLines(contents);
    std::vector<std::string> lines;
    for (std::string line; std::getline(csvLines, line);) {
        lines.push_back(line);
    }
    ASSERT_EQ(lines.size(), 9U);
    EXPECT_NE(lines[1].find("accounting,"), std::string::npos);
    EXPECT_NE(lines[1].find(",baseline,1,"), std::string::npos);
    EXPECT_NE(lines[2].find(",snap-sentinel,1,"), std::string::npos);
    EXPECT_NE(lines[3].find("performance,"), std::string::npos);
    EXPECT_NE(lines[3].find(",baseline,1,"), std::string::npos);
    EXPECT_NE(lines[5].find(",snap-sentinel,2,"), std::string::npos);
    EXPECT_NE(lines[6].find(",baseline,2,"), std::string::npos);
    csv.close();
    EXPECT_TRUE(std::filesystem::exists(csvPath));

    std::filesystem::remove(csvPath, error);
}

}  // namespace
}  // namespace snapsentinel::benchmark
