# Repository Guidelines

## Project Structure & Module Organization
Headers live under `include/much/`, implementation files sit in `impl/`, executables stay in `src/`, and `data/` holds MNIST IDX files plus generated weights (gitignored except `.gitkeep`). When adding functionality, drop the public API in `include/much/`, the implementation in `impl/`, and list the new source in the `MUCH_IMPL_SOURCES` section of `CMakeLists.txt` so it becomes part of the `much_core` static library that powers `src/main.c`.

## Build, Test, and Development Commands
- `cmake -B build -S . -G Ninja` — configure the project and locate BLAS/Accelerate.
- `ninja -C build` — build `libmuch_core.a` plus the `much` demo.
- `./build/much` — run the MNIST training loop using the files in `data/`.
- `./run_mnist.sh` — helper that rebuilds with curated `OPENBLAS_NUM_THREADS` before running.

## Coding Style & Naming Conventions
Stick with C11, two-space indentation, brace-on-same-line functions, and snake_case identifiers (`tensor_f32_randn`). Public structs end with `_t`, headers use `#pragma once`, and every include uses the `much/` prefix (`#include "much/tensor.h"`). Keep helper functions `static` inside their `impl/` translation unit, and add concise comments only around math-heavy or non-obvious ownership logic.

## Testing Guidelines
`./run_mnist.sh` doubles as the regression test: it configures, builds, and executes the demo end-to-end. Capture the training log plus final accuracy (currently ~90%+) for every submission. When touching kernels or optimizers, add temporary assertions or deterministic micro-tests in `src/` or a scratch executable, then remove them once confidence is restored. Call out any environment variables or compiler flags needed to reproduce your results.

## Commit & Pull Request Guidelines
Use concise, imperative summaries (`tensor: guard matmul dims`) under 72 characters. Provide a short body covering motivation, performance impact, and test evidence. Pull requests should mention structural changes (new headers, sources moved between include/impl/src, or data layout shifts), enumerate the commands you ran (`cmake`, `ninja`, `run_mnist.sh`), and attach updated accuracy/throughput numbers or screenshots whenever behavior changes.

## Data & Configuration Tips
Always place MNIST assets and `weights.bin` inside `data/`, using the relative paths baked into `src/main.c`. Avoid committing raw data by keeping everything under that directory. If you switch BLAS providers, ensure `find_package(BLAS)` can discover them (set `BLAS_LIBRARIES`, `BLAS_INCLUDE_DIRS`, or `PKG_CONFIG_PATH`) and document any non-default threading or deterministic settings so other contributors can reproduce your environment quickly.
