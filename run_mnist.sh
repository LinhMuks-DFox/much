#!/bin/bash
OPENBLAS_NUM_THREADS=4 ninja -C build && OPENBLAS_NUM_THREADS=4 ./build/much
