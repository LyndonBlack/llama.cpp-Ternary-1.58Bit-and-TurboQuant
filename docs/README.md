# Ternary 1.58-bit + TurboQuant fork notes

This branch combines three pieces of work on top of current upstream `llama.cpp`:

1. PrismML Q2_0 / ternary model support for 1.58-bit-style GGUF models.
2. TurboQuant KV cache support, including `turbo2_0`, `turbo3_0`, and `turbo4_0` ggml types.
3. CUDA FlashAttention vector-path optimizations for the known-good `K=turbo4_0`, `V=turbo3_0` cache combination.

Repository: <https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>

## Branch and checkpoint

- Working branch: `turboternary`
- Known-good code checkpoint: `turboternary-q2-turbokv-96tps-2026-05-07`
- Pre-TurboKV Q2_0 checkpoint: `q2-baseline-before-turbokv-2026-05-07`

## Build

CUDA release build used for validation:

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS='-static-libstdc++ -static-libgcc' -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++ -static-libgcc' && make -j$(nproc)
```

The static libstdc++/libgcc flags avoid the local CUDA/libstdc++ linker issue seen during integration.

## Recommended Bonsai command

```bash
~/AI/ternaryturboquant/llama.cpp/build/bin/llama-cli -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf -p "Write a detailed, structured technical essay about where snails live, covering habitats, climate, food sources, predators, reproduction, and ecological roles. Continue in clear paragraphs." -n 4096 -c 65535 -s 12345 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1.0 --ignore-eos --no-display-prompt --flash-attn on -ngl 40 -ctk turbo4_0 -ctv turbo3_0
```

Observed result on RTX 3070 Ti: about `96.5-99.6 t/s` generation depending on run/tooling, matching the original Prism TurboKV checkpoint.

## Recommended large Qwen MoE command

```bash
~/AI/ternaryturboquant/llama.cpp/build/bin/llama-cli -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf -p "Write a detailed, structured technical essay about where snails live, covering habitats, climate, food sources, predators, reproduction, and ecological roles. Continue in clear paragraphs." -ngl 99 --n-cpu-moe 35 --no-mmap --mlock -ctk turbo4_0 -ctv turbo3_0 --no-warmup -s 12345 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1.0 --ignore-eos --no-display-prompt --flash-attn on -n 4096 -c 256000
```

Observed result on RTX 3070 Ti + CPU MoE offload: around `43-45 t/s`, with no perceptible slowdown as the response grows in the tested range.

## KV cache guidance

Known-good path:

```text
-ctk turbo4_0 -ctv turbo3_0 --flash-attn on
```

Other supported cache types parse and run in many paths (`f16`, `q8_0`, `q4_0`, `turbo2_0`, `turbo3_0`, `turbo4_0`), but the CUDA FlashAttention path has been tuned and validated specifically for `turbo4_0` K plus `turbo3_0` V.

## Current limitations / future work

The current branch is a stable baseline. Future experimental work should happen on separate branches.

Good candidates:

- InnerQ scale graph plumbing: likely quality-oriented rather than speed-critical.
- Turbo2/Turbo3 K LUT scoring: useful for testing more aggressive K compression modes.
- Adaptive / layer-wise KV cache policies: potentially useful for large MoE models.
- MLA/ISWA Turbo graph paths: broader Qwen/DeepSeek coverage, more invasive than the current MHA-focused path.
- More formal quality testing: perplexity or fixed prompt comparisons across KV types.

## Raw benchmark logs

See `benchmark-results/2026-05-07-turboternary/` for the raw outputs captured during the documentation checkpoint.
