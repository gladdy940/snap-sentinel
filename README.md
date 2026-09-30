# Snap-Sentinel — Zero-Latency Edge Vision Engine

> **Concept:** Reduce avoidable staging copies and scratch-buffer allocations
> between camera-frame ownership and edge-inference tensor preparation.

Snap-Sentinel is a C++20 edge-vision pipeline prototype. Its host reference
experiment compares a conventional staged preprocessing path with a path that
converts pixels directly from an owned frame buffer into a separately
allocated inference tensor. The title describes the project goal; it is not a
claim of measured zero latency or hardware zero-copy.

## The problem: unnecessary data movement before inference

An edge-vision pipeline may move a captured image through one or more CPU
staging buffers before preparing an inference tensor. Those staging steps can
add memory traffic and allocation pressure before inference begins. Snap-Sentinel
explores reducing this staging work while preserving the same input semantics
and inference result.

## Compared data paths

### Conventional baseline

For each gray8 frame, the reference baseline:

1. Copies the frame pixels into a byte preprocessing buffer.
2. Normalizes those pixels into an intermediate float buffer.
3. Copies the float buffer into separately allocated float32 tensor storage.
4. Sends the tensor to the mock inference engine.

### Snap-Sentinel

The reference Snap-Sentinel path:

1. Passes the captured frame through the asynchronous pipeline by moving frame
   ownership; the queue handoff does not copy the image payload.
2. Reads the gray8 pixels from the owned frame buffer, normalizes them, and
   writes them directly into separately allocated float32 tensor storage.
3. Sends that tensor to the same mock inference engine.

The paths share dimensions, deterministic input generation, normalization,
tensor shape/layout, inference implementation, and output handling. The
Snap-Sentinel preparer still writes each normalized tensor element (using a
per-element `memcpy` in the current implementation); it does not reuse a tensor
or frame-buffer pool.

## Target Snapdragon architecture

The intended hardware data path is:

```text
Camera / Spectra ISP
        |
        v
Hardware-accessible camera buffer
        |
        v
AHardwareBuffer / dma_buf
        |
        v
Tensor preparation
        |
        v
Qualcomm QNN
        |
        v
Hexagon HTP / NPU
```

This is a **target architecture**, not the currently validated host path. The
host `Buffer` is backed by `std::vector<std::byte>`, camera input is generated
by `MockCameraCapture`, and inference uses `MockInferenceEngine`. Connecting a
real Spectra ISP buffer to QNN, validating memory import, and executing on
Hexagon HTP/NPU remain future hardware work.

## Current implementation architecture

```text
MockCameraCapture
        |
        v
Frame + shared Buffer ownership
        |
        v
AsyncFrameLoop (bounded queue; capture and dispatch workers)
        |
        v
FramePipeline
        |
        v
Gray8 -> normalized float32 NHWC tensor
        |
        v
MockInferenceEngine -> completion callback
```

The portable interfaces separate capture, tensor preparation, and inference.
The current mock inference call scans the prepared tensor synchronously. The
baseline and Snap-Sentinel paths are experiment implementations; the baseline
is a representative staged path, not a claim that all production camera
pipelines use that exact sequence.

## Benchmark methodology

The data-path experiment uses deterministic gray8 frames generated from the
frame ID and pixel index. Both paths use the same frame count, dimensions,
warm-up count, preprocessing semantics, mock inference engine, and output
validation. Each measured iteration alternates path order.

It has two distinct measurement modes:

- **Accounting:** enables optional counters for the defined staging-copy and
  allocation categories.
- **Performance:** disables those optional counter updates. It runs the same
  underlying path algorithms and measures pipeline latency and FPS.

For performance measurements:

- **Pipeline latency** is measured from the captured frame timestamp to
  inference completion. It includes queueing/scheduling, preprocessing, and
  mock inference.
- The mock camera allocates and fills the frame buffer **before** that
  timestamp, so this work is outside the pipeline-latency interval.
- **FPS** is measured as processed frames over wall-clock time from pipeline
  start through all completions and worker shutdown.

The CSV records per-iteration rows, measurement mode, build configuration,
workload dimensions, performance values, and accounting fields. Results below
are host-side reference observations for the unchanged workload: 1,000 frames
per run, 3 iterations, 20 warm-up frames, 64×48 gray8, normalized float32
NHWC, Release build.

### Validated accounting results

The values below are totals across the 3,000 measured frames:

| Instrumented category | Conventional baseline | Snap-Sentinel |
|---|---:|---:|
| Full-frame staging copies | 3,000 | 0 |
| Full-tensor staging copies | 3,000 | 0 |
| Instrumented staging bytes copied | 46.08 MB | 0 |
| Captured-frame payload allocations | 3,000 | 3,000 |
| Scratch/staging allocations | 6,000 | 0 |
| Tensor payload allocations | 3,000 | 3,000 |
| Logical inference output equivalence | PASS | PASS |

**46.08 MB is instrumented staging-copy traffic, not total memory traffic.**
It is the sum of one 3,072-byte frame staging copy and one 12,288-byte float
tensor staging copy per frame, over 3,000 frames. The accounting counters do
not measure every read, write, store, compiler/runtime operation, or memory
transfer. Likewise, the allocation counters describe only the named
instrumented categories; they are not total heap-allocation counts.

“Zero” in the Snap-Sentinel staging columns means zero **instrumented
full-payload staging copies** in this experiment. It does not mean no tensor
stores, no per-element `memcpy`, no memory movement, or hardware zero-copy.
Output equivalence is checked by comparing frame IDs, inference scores, and
success status across the two paths.

### Performance interpretation

The measured host performance is workload- and host-dependent, and the
comparison has shown run-to-run variability, including changes in which path
is faster. The experiment does not establish a stable FPS improvement; latency
or throughput superiority is not a correctness requirement and no favorable
result is assumed.

## Build, test, and run

Requirements: CMake 3.20+, a C++20 compiler, and GoogleTest when testing is
enabled. The host build does not require the Android NDK or Qualcomm QNN SDK.

Configure a clean Release host build and compile with two jobs:

```sh
cmake -S . -B build/host-audit-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build/host-audit-clean -j2
```

Run tests:

```sh
ctest --test-dir build/host-audit-clean --output-on-failure
```

Run the host data-path experiment with the validated workload:

```sh
./build/host-audit-clean/snapsentinel_runner \
  --frames 1000 \
  --iterations 3 \
  --width 64 \
  --height 48 \
  --warmup-frames 20 \
  --output-csv build/host-audit-clean/data_path_experiment.csv
```

The experiment CSV is generated output and should be directed to a build or
temporary directory rather than committed as source documentation.

## Validation status

The documented host validation completed with:

- Clean Release configuration and build: **passed**.
- CTest: **1/1 test executable passed**.
- GoogleTest: **18 passed, 0 failed**.
- `git diff --check`: **passed**.
- Baseline/Snap-Sentinel output equivalence: **PASS**.

## Limitations and next steps

- This is a **host-side reference experiment** using mock capture and mock
  inference, not a real camera-to-inference device benchmark.
- It does **not** prove literal hardware zero-copy.
- It does **not** prove a reduction in total memory traffic; only the specified
  instrumented staging-copy categories are counted.
- It does **not** prove Snapdragon, QNN, Hexagon HTP, or NPU performance.
- It does **not** prove a stable FPS improvement.
- Snapdragon hardware validation is future work: integrate a real camera/ISP
  buffer, validate AHardwareBuffer/dma_buf import and tensor preparation with
  the supported QNN SDK, then measure end-to-end behavior on target devices.

No Snapdragon FPS, NPU performance, power, or thermal claim should be inferred
from this host mock benchmark.
