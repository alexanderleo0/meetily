# GigaAM v3 E2E RNNT in Meetily

This fork adds **GigaAM v3 E2E RNNT INT8** as a separate local Russian speech
recognition provider. Parakeet and Whisper remain available. The first supported
GigaAM target is **macOS Apple Silicon**. Inference is native Rust + ONNX Runtime
on CPU (four threads); Python and an external transcription service are not needed.

## Build

Install Apple's Command Line Tools, Rust, CMake, Node.js and pnpm. Then, from the
checkout root:

```sh
./scripts/build-macos-gigaam.sh
```

The application is produced at `target/release/bundle/macos/Meetily GigaAM.app`.
It has its own identifier, `com.alexanderleo0.meetily`, and data directory, so it
can coexist with upstream Meetily. The local build is ad-hoc signed; notarization
and automatic update artifacts are not configured for this fork.

The script builds the llama-helper and FFmpeg sidecars. With Command Line Tools
alone it uses the narrowly scoped `scripts/macos-clt/xcodebuild` adapter to compile
the pinned cidre dependency's Objective-C static libraries with Apple's clang.
Full Xcode builds use the real xcodebuild. Only needed cidre features are enabled.
Whisper retains Metal/CoreML. The unused HTTPS client in llama-helper is disabled
at CMake configuration time to avoid an OpenSSL 4 incompatibility in its vendor code.
If retrying a helper build first configured without this setting, clear that
package's build cache with `cargo clean -p llama-cpp-sys-2 --release` first.

## Use

Choose GigaAM during first-run setup, or open Transcript Settings and choose
**GigaAM v3 (Russian · Local · Apple Silicon)**. Download the model, then click
**Use GigaAM**. Settings are saved only after integrity verification and a successful
model load. The selected provider persists across restarts.

The download is approximately 227 MB. Every file has a pinned revision, byte size
and SHA256 in `frontend/src-tauri/src/gigaam_engine/models.lock.json`. Files are
streamed into `.part` files, checked, then renamed atomically. Cancellation removes
the incomplete file; retry reuses verified completed files and replaces corrupt ones.
Model files live in the application's data directory under
`models/gigaam-v3-e2e-rnnt-int8`. Network access is needed only to download public
assets; audio is not uploaded by this engine.

Start a meeting as usual. GigaAM uses the existing VAD speech queue, transcript
events and recording timestamps. It emits finalized phrase text, not incremental
word hypotheses. Language is Russian; Whisper's saved language/translation choice
does not affect GigaAM.

In **Import Audio** and **Retranscribe**, the downloaded GigaAM model appears beside
available Parakeet and Whisper models. Both batch paths use the same GigaAM engine,
existing decoding/resampling/VAD, database persistence, cancellation and timestamp
handling. Batch completion and recording shutdown unload GigaAM when safe.
Confidence is unavailable; timestamps are phrase boundaries, not word alignment.
On macOS, model loading and inference hold a scoped user-initiated activity so
background work is not treated as idle app work. The activity ends even on errors
and permits normal system idle sleep. See [Apple’s activity guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheAppLevel.html).

## Native verification

The original Python prototype remains outside this checkout in the parent directory.
A source-only reference copy is in `docs/onnx-comparison`; its audio, results, model
cache and Python environment are intentionally absent from Git.

`gigaam_verify` compares native decoding on the prototype's shared VAD boundaries.
Provide mono 16 kHz f32 little-endian PCM and its manifest:

```sh
cargo run --release -p meetily --example gigaam_verify -- \
  /path/to/model-directory /private/input.f32 /private/manifest.json /private/native.json
```

`gigaam_smoke` exercises the actual Tauri engine validation, live worker/event path,
full file import and full retranscription. It uses a separate validation application
data directory and creates a validation meeting in `~/Movies/meetily-recordings`.
It does not activate microphone or system capture. The recording must be at least
120 seconds long with speech at 100–120 seconds for the live excerpt check.

```sh
cargo run --release -p meetily --example gigaam_smoke -- \
  /private/meeting.m4a /path/to/model-directory /private/smoke.json
```

Append `--download` to download and verify the locked assets through the application's
real downloader instead of copying a local model. Private transcripts are written
only to the meeting directory and explicitly supplied outputs. On CLT-only machines,
use the same PATH adjustment as the build script for direct Cargo commands.

## Sources and licenses

- GigaAM: https://github.com/salute-developers/GigaAM
- ONNX conversion: https://huggingface.co/istupakov/gigaam-v3-onnx
- Decoder/preprocessor reference: https://github.com/istupakov/onnx-asr (0.12.0, MIT).
  The upstream MIT notice is included in `gigaam_engine/ONNX-ASR-LICENSE` and beside
  the redistributed preprocessing asset. Speech model weights remain external;
  consult the upstream GigaAM model license for their terms.
