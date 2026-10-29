#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This build script targets macOS Apple Silicon.' >&2
  exit 1
fi
export NEXT_TELEMETRY_DISABLED=1
export CMAKE_PROJECT_INCLUDE="$PWD/scripts/cmake/local-llama.cmake"
# cidre's small Objective-C libraries do not require the Xcode GUI; use clang
# from Command Line Tools when the full xcodebuild executable is unavailable.
if ! /usr/bin/xcodebuild -version >/dev/null 2>&1; then
  export PATH="$PWD/scripts/macos-clt:$PATH"
fi
cargo build --locked --release -p llama-helper --features metal
mkdir -p frontend/src-tauri/binaries
cp target/release/llama-helper frontend/src-tauri/binaries/llama-helper-aarch64-apple-darwin
cd frontend
pnpm install --frozen-lockfile
pnpm exec tauri build --bundles app
