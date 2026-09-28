cmake_minimum_required(VERSION 3.20)

project(SnapSentinel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
include(CTest)

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

add_library(snapsentinel_core STATIC
    src/core/DmaBufferManager.cpp
    src/core/AsyncNpuEngine.cpp
    src/pipeline/BenchmarkRunner.cpp
)
target_include_directories(snapsentinel_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(snapsentinel_core SYSTEM PUBLIC "${QNN_INCLUDE_DIR}")
target_link_libraries(snapsentinel_core PUBLIC
    Threads::Threads
    "${QNN_HTP_LIBRARY}"
    "${QNN_SYSTEM_LIBRARY}"
    ${CMAKE_DL_LIBS}
)

add_executable(snapsentinel_runner src/main.cpp)
target_link_libraries(snapsentinel_runner PRIVATE
    snapsentinel_core
    Threads::Threads
)

if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_executable(snapsentinel_tests
        tests/test_main.cpp
        tests/benchmark_runner_test.cpp
    )
    target_link_libraries(snapsentinel_tests PRIVATE
        snapsentinel_core
        Threads::Threads
        GTest::gtest
    )
    add_test(NAME snapsentinel_tests COMMAND snapsentinel_tests)
endif()

foreach(target snapsentinel_core snapsentinel_runner snapsentinel_tests)
    if(TARGET "${target}")
        target_compile_options("${target}" PRIVATE
            "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>"
        )
    endif()
endforeach()
