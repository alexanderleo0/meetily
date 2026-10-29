# GigaAM integration validation

Validated on macOS Apple Silicon on 2026-10-29. The supplied private recording is
1714 seconds long. Audio and transcripts are not part of this repository.

## Decoder correctness

The native decoder completed all 165 speech segments from the prototype's shared
VAD manifest. A separate Python reference using the same pinned weights, ONNX
preprocessor and ONNX Runtime 1.22.0 produced **165/165 exact text matches**.

The original prototype used ONNX Runtime 1.31.1 and onnx-asr's default NumPy
preprocessor. Against that saved result, 120/165 entire phrases match exactly.
Changing only Python to the ONNX preprocessor at Runtime 1.31.1 gives 122/165.
Those comparisons are not accuracy scores: different floating-point preprocessing
and INT8 runtime kernels can change greedy token choices. No manual reference was
provided, so WER or an accuracy winner is not claimed.

The native shared-segment run took 47.92 seconds, while compilation was running.
This is a functional test, not a controlled performance benchmark or UI latency
measurement. The original prototype's timings remain in `onnx-comparison/RESULTS.md`.

## Build and component checks

- Next.js production export and TypeScript checks pass.
- Native ARM64 release binary, FFmpeg and Metal llama-helper sidecars build.
- macOS `.app` bundle is ad-hoc signed. No developer signing identity or notarization
  credentials are required for this local build.
- Native tests pass for manifest completeness, SHA256/size corruption detection,
  empty/silent audio and rejection of non-finite samples.
- UI inspection verified first-run GigaAM selection, model download progress,
  cancellation, retry, completion and the Use GigaAM action.
- The real downloader fetched all locked assets into a fresh validation data directory
  and verified their hashes. A transient GitHub connection failure was recovered by
  retrying; already verified weight files were reused.

The end-to-end run uses the real Tauri configuration API, live transcription worker,
transcript events, import pipeline, database and retranscription pipeline. It feeds
a recorded speech excerpt into the live queue; hardware capture permissions are
left to the user. The microphone and system audio devices are not activated by
this automated verification.

## Full application pipeline result

- Saved configuration was round-tripped through the actual Tauri API with provider `gigaam`.
- Live VAD queue/worker emitted 1 finalized transcript event from the supplied recording.
- Full import: **76 nonempty transcript segments**, persisted to SQLite and the meeting's `transcripts.json`.
- Full retranscription: **76 segments**, with **identical text** to the imported transcript.
- Imported audio copy matches the original file's SHA256.
- Recording duration reported by the app: 1714.05 seconds.
- Import wall time: 75.17 seconds; retranscription wall time: 163.48 seconds.
  These include decoding, resampling, VAD, model loading and persistence, and ran
  alongside build work. They are not controlled ASR benchmarks.

A first headless run was heavily deprioritized while the application was invisible.
Model loading and inference now use a scoped macOS user-initiated activity, allowing
normal idle sleep, and the complete background validation run passes.

The application's batch VAD uses 2-second silence redemption and up to 25-second
segments with silence-aware splitting, so its 76 segments differ from the Python
prototype's 165 shared comparison segments. Playback timestamps remain relative
to the original recording.
