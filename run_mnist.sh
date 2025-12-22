#!/bin/bash
OPENBLAS_NUM_THREADS=23 ninja -C build && OPENBLAS_NUM_THREADS=4 ./build/much
