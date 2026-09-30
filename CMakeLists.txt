cmake_minimum_required(VERSION 3.20)

project(SnapSentinel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SNAP_SENTINEL_ENABLE_QNN "Build the Android Snapdragon QNN backend" OFF)

find_package(Threads REQUIRED)
include(CTest)

if(WIN32)
    set(SNAP_SENTINEL_PLATFORM_METRICS
        platform/windows/benchmark_metrics.cpp
    )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
    set(SNAP_SENTINEL_PLATFORM_METRICS
        platform/linux/benchmark_metrics.cpp
    )
else()
    message(FATAL_ERROR "Snap-Sentinel supports Windows, Linux, and Android builds")
endif()

add_library(snapsentinel_core STATIC
    src/core/buffer.cpp
    src/core/frame.cpp
    src/inference/tensor_preparer.cpp
    src/inference/mock_inference_engine.cpp
    src/pipeline/camera_capture.cpp
    src/pipeline/async_frame_loop.cpp
    src/pipeline/frame_pipeline.cpp
    src/benchmark/benchmark_runner.cpp
    src/benchmark/data_path_experiment.cpp
    ${SNAP_SENTINEL_PLATFORM_METRICS}
)
target_include_directories(snapsentinel_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_definitions(snapsentinel_core PRIVATE
    SNAP_SENTINEL_BUILD_CONFIGURATION="$<CONFIG>"
)
target_link_libraries(snapsentinel_core PUBLIC Threads::Threads)

add_executable(snapsentinel_runner app/main.cpp)
target_link_libraries(snapsentinel_runner PRIVATE snapsentinel_core)

if(SNAP_SENTINEL_ENABLE_QNN)
    if(NOT ANDROID)
        message(FATAL_ERROR "SNAP_SENTINEL_ENABLE_QNN requires an Android ARM64 toolchain")
    endif()
    if(NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "SNAP_SENTINEL_ENABLE_QNN requires CMAKE_ANDROID_ARCH_ABI=arm64-v8a")
    endif()

    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Path to the Qualcomm QNN SDK")
    if(NOT QNN_SDK_ROOT)
        message(FATAL_ERROR "QNN_SDK_ROOT must point to the Qualcomm QNN SDK")
    endif()

    find_path(QNN_INCLUDE_DIR
        NAMES QnnInterface.h
        PATHS "${QNN_SDK_ROOT}/include/QNN"
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_library(QNN_HTP_LIBRARY
        NAMES QnnHtp
        PATHS "${QNN_SDK_ROOT}/lib/aarch64-android"
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_library(QNN_SYSTEM_LIBRARY
        NAMES QnnSystem
        PATHS "${QNN_SDK_ROOT}/lib/aarch64-android"
        NO_DEFAULT_PATH
        REQUIRED
    )

    add_library(snapsentinel_qnn STATIC
        platform/android/qnn/qnn_runtime.cpp
        platform/android/qnn/qnn_inference_engine.cpp
        platform/android/qnn/qnn_buffer_import.cpp
    )
    target_include_directories(snapsentinel_qnn
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/platform/android/qnn"
    )
    target_include_directories(snapsentinel_qnn SYSTEM PUBLIC "${QNN_INCLUDE_DIR}")
    target_link_libraries(snapsentinel_qnn PUBLIC
        snapsentinel_core
        "${QNN_HTP_LIBRARY}"
        "${QNN_SYSTEM_LIBRARY}"
        ${CMAKE_DL_LIBS}
    )
endif()

if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_executable(snapsentinel_tests
        tests/test_main.cpp
        tests/core/buffer_test.cpp
        tests/pipeline/frame_pipeline_test.cpp
        tests/pipeline/end_to_end_test.cpp
        tests/inference/tensor_test.cpp
        tests/benchmark/benchmark_runner_test.cpp
        tests/benchmark/data_path_experiment_test.cpp
        tests/platform/metrics_test.cpp
    )
    target_link_libraries(snapsentinel_tests PRIVATE
        snapsentinel_core
        GTest::gtest
    )
    add_test(NAME snapsentinel_tests COMMAND snapsentinel_tests)
endif()

foreach(target snapsentinel_core snapsentinel_runner snapsentinel_tests snapsentinel_qnn)
    if(TARGET "${target}")
        target_compile_options("${target}" PRIVATE
            "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>"
            "$<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>"
        )
    endif()
endforeach()
