# Integration summary — Ternary 1.58-bit + TurboQuant

## Purpose

This branch is a clean current-upstream fork that combines:

- PrismML Q2_0 / ternary GGUF model support.
- TurboQuant KV cache types and cache construction.
- CUDA FlashAttention vector-path support for TurboKV.
- Proven CUDA tuning from the Prism checkpoint.

## Main commits

Q2_0 / ternary baseline:

- `b525f1510 Add Q2_0 quantization: type definition and CPU backend`
- `63197db69 Add Q2_0 Metal backend`
- `36084f24f cuda: Q2_0`

TurboQuant baseline and CUDA path:

- `e3225e5cf Add TurboQuant KV cache baseline`
- `f787cea80 Add Turbo KV FlashAttention vec routing`
- `cb4bc74e2 Tune Turbo KV FlashAttention CUDA path`
- `5c136e3ba Allow Turbo KV cache types in llama-bench`

## Validated models

- `Ternary-Bonsai-8B-Q2_0.gguf`
  - Q2_0 model loading and CUDA inference validated.
  - `-ctk turbo4_0 -ctv turbo3_0` validated up to ~65k context.

- `Qwen3.6-35B-A3B-Q5_K_M.gguf`
  - Large MoE CPU/GPU split path validated with `--n-cpu-moe 35` and 256k context setting.
  - Performance matched or slightly exceeded the older `llama-cpp-turboquant` build in local tests.

## Upstream status

At the time of integration, `ggml-org/llama.cpp` and the renamed GitHub fork did not contain native TurboQuant source implementation entries for `turbo4_0`, `turbo3_0`, `QK_TURBO`, or `GGML_TYPE_TURBO`. The working TurboQuant reference remained TheTom / Prism-derived code path.

## Recommended stable branch policy

Treat `turboternary` as the stable integration branch. Further optimization work should branch from it, for example:

- `turboternary-innerq`
- `turboternary-adaptive-kv`
- `turboternary-turbo-lut`
- `turboternary-mla-iswa`

## Known future work

1. InnerQ scale plumbing for quality-oriented tuning.
2. Turbo2/Turbo3 K LUT scoring and template coverage.
3. Adaptive/layer-wise KV cache selection.
4. MLA/ISWA Turbo graph paths for broader architecture coverage.
5. Formal quality/perplexity tests across KV cache modes.
