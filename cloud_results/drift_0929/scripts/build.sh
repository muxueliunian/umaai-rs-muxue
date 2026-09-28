#!/bin/bash
export RUSTUP_TOOLCHAIN=1.96.0 CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_DEBUG=false
python3 <WORK>/timed.py <ROOT>/$1 cargo build --release -j 2 -p umasim --features "cli onnx" --bin ramen_space_bench > <WORK>/logs/build_$1.log 2>&1
