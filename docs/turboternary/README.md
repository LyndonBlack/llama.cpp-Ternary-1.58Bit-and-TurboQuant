# Ternary 1.58-bit + TurboQuant fork notes

This branch combines four pieces of work on top of current upstream `llama.cpp`:

1. **PrismML Q2_0 / ternary model support** for 1.58-bit-style GGUF models.
2. **TurboQuant KV cache support**, including `turbo2_0`, `turbo3_0`, `turbo4_0`, and `turbo6_0` ggml types.
3. **CUDA FlashAttention** vector-path optimizations.
4. **SCJedi Entropy-Adaptive KV Cache** — per-head entropy-informed compression, available as Path B (per-layer mixed-precision K types).

Repository: <https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>

Research inspirations:
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026)
- [SCJedi/entropy-adaptive-kv-cache](https://github.com/SCJedi/entropy-adaptive-kv-cache)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — follow-on research:
  - V compression is free (confirmed independently across Metal, CUDA)
  - All quality degradation comes from K compression (asymmetric K/V is the right approach)
  - Boundary layers (first 2 + last 2) are disproportionately sensitive

## Build

CUDA release build used for validation:

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS='-static-libstdc++ -static-libgcc' -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++ -static-libgcc' && make -j$(nproc)
```

The static libstdc++/libgcc flags avoid the local CUDA/libstdc++ linker issue seen during integration.

---

## Entropy-Guided KV Cache Compression (Path B)

**What it does:** Uses per-head attention entropy to decide which transformer layers can use `turbo4_0` (4-bit) K vs `q8_0` (8-bit) K. Low-entropy (sink/focused) heads tolerate aggressive compression; high-entropy (diffuse) heads keep full precision.

**Architecture:** The entropy profile is calibrated once per model, saved as a JSON file. At server startup, it's loaded and a per-layer callback (`entropy_layer_k_cb`) is set that returns the appropriate `ggml_type` for each layer based on its mean entropy relative to the median.

**Flag:** `--entropy-profile <path>` activates Path B. `--entropy-prune-ratio <float>` controls aggressiveness (>2.0 uses TURBO2_0 for very low entropy layers).

**The key insight:** Path B keeps the *full requested context size* — it saves memory through mixed precision, not by reducing the context window. No more silent context halving.

### Calibration Pipeline

```bash
# Run entropy calibration on any model
./build/bin/llama-entropy-calibrate -m ~/AI/models/<model>.gguf \
  -ngl 99 --n-cpu-moe <n_layers> -c 64
```

Saves `entropy_profile.json` with per-head entropy data. Calibration requires flash-attn to be automatically disabled during capture, then re-enabled.

### YARN Compatibility

YARN (extending position encoding beyond trained context) is complementary. YARN extends what the model can *understand*, Path B frees memory to accommodate more tokens. Entropy profiles are transferable to extended context because head specialization is an architectural property, not position-dependent.

---

## MoE Models with CPU Expert Offloading

**These are the gold standard for limited VRAM.** With `--n-cpu-moe <n_layers>`:
- The full model loads in system RAM (needs 32 GB+)
- Only active expert weights are computed on CPU
- KV cache + small tensors live in GPU VRAM
- CodeNeedle testing confirms identical quality to full-GPU at any reasonable context size

Example for Qwen3.6 35B A3B (30B total, ~3B active per token):
```
-ngl 99 --n-cpu-moe 39 --no-mmap
```

---

## Recommended Command Lines

### Bonsai 8B (q2, small test model)
```bash
llama-cli -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf \
  -p "Your prompt here" -n 4096 -c 65535 \
  -s 12345 --temp 0 --top-k 1 --top-p 1 \
  --flash-attn on -ngl 40 \
  -ctk q8_0 -ctv turbo3_0
```

### Qwen3.6 35B A3B (PRIMARY — text + vision, ~35-40 t/s)
```bash
llama-server \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  --mmproj ~/AI/models/mmproj-Qwen3.6-35B-A3B-F16.gguf \
  --no-mmproj-offload \
  --alias qwen3.6-35b-a3b \
  -ngl 99 --n-cpu-moe 39 --no-mmap \
  --ctx-size 256000 --flash-attn on \
  -ctk q8_0 -ctv turbo3_0 \
  --host 127.0.0.1 --port 8080 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen.json \
  --entropy-prune-ratio 2.0
```

### Qwen3-VL-30B A3B (vision MoE, comparison only)
```bash
llama-server \
  -m ~/AI/models/Qwen_Qwen3-VL-30B-A3B-Instruct-Q5_K_L.gguf \
  --mmproj ~/AI/models/mmproj-Qwen_Qwen3-VL-30B-A3B-Instruct-f16.gguf \
  --no-mmproj-offload \
  --alias qwen3-vl-30b \
  -ngl 99 --n-cpu-moe 48 --ctx-size 131072 --flash-attn on \
  -ctk q8_0 -ctv turbo3_0 \
  --host 127.0.0.1 --port 8081 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen3vl.json \
  --entropy-prune-ratio 2.0
```

---

## CodeNeedle Validation Results

[CodeNeedle](https://github.com/codeneedle) is a positional recall benchmark: given a real code file in context, reproduce the first 20 lines of a named function.

### http_server corpus (~14K tokens, 11 targets)

| Model | Config | Pass | Matched | Hallucinated |
|-------|--------|:----:|:-------:|:------------:|
| Bonsai 8B Q2 | q8 K + turbo3 V | 11/11 | 218/220 | 0 |
| **Qwen3.6 35B A3B** | **Path B, q8 K + turbo3 V** | **11/11** | **220/220** | **0** |
| Qwen3-VL-30B A3B | Path B, q8 K + turbo3 V | 7/11 | 125/220 | 40 |

### jQuery corpus (~80K tokens, 16 targets)

| Model | Config | Pass | Matched | Hallucinated | Speed |
|-------|--------|:----:|:-------:|:------------:|:-----:|
| Bonsai 8B Q2 | 32K ctx — too small for this test | N/A | N/A | N/A | N/A |
| **Qwen3.6 35B A3B** | **Path B, 256K ctx** | **15/16** | **282/320** | **39** | **~40 t/s** |
| Qwen3-VL-30B A3B | Path B, 128K ctx | 13/16 | 247/320 | 221 | ~26 t/s |

**Qwen3.6 35B A3B is the clear winner** — perfect on small corpus, strong on large, fast, and vision-capable.

---

## Entropy Profiles (Pre-Calibrated)

| Model | File | Heads | Mean Entropy | CV |
|-------|------|:----:|:------------:|:--:|
| Bonsai 8B | `entropy_profile_bonsai.json` | 1152 | 0.620 | 0.58 |
| Qwen3.6 35B A3B | `entropy_profile_qwen.json` | 640 | 0.620 | 0.55 |
| Qwen3-VL-30B A3B | `entropy_profile_qwen3vl.json` | 1536 | 0.468 | 0.42 |
| GLM-4.6V-Flash 9B | `entropy_profile_glm.json` (archived) | 1280 | 0.502 | 0.37 |

---

## Recommended KV Cache Types

| Type | K | V | Notes |
|------|---|----|-------|
| **Stable** | **q8_0** | **turbo3_0** | Production default — all K at 8-bit, V at 3.125-bit |
| Aggressive q8 path | q8_0 | turbo2_0 | V compression is "free" per research — try this for +36% V savings |
| Max speed (Bonsai) | turbo4_0 | turbo3_0 | Proven for Bonsai on RTX 3070 Ti |
| Path B mixed | entropy-guided mix | turbo3_0 | Layers with entropy below median get turbo4_0, rest keep q8_0 |

---

## Future Research Directions

### From TheTom/turboquant_plus (to explore)
- **Block size optimization**: 5.12× compression potential
- **Sparse V dequant**: +22.8% decode speed
- **Boundary layer tuning**: Protecting first 2 + last 2 layers at higher precision recovers 37-91% of quality gap from aggressive K compression
- **V at turbo2**: TheTom's independent validation confirms this is free when K precision is maintained

### True per-head granularity
Split K tensor within a layer (the `k_extra` field is ready in our code). Beyond layer-level types, do head-level budget allocation.

### Dynamic eviction
Use `prune_by_importance` with actual attention data from the FA path — prune KV entries per-head based on real-time attention scores rather than static entropy profiles.

---

## Future Project: "Mostlysane Local AI"

**Vision:** A web app where users enter their hardware specs (GPU VRAM, system RAM) and get a complete working local AI configuration — optimized for MoE models with shared experts.

**Stack:** Static web app on GitHub Pages for the config generator, HuggingFace Spaces for any server-side compute.

**Features:**
- Pre-curated model list (tested on our RTX 3070 Ti 8GB)
- Manual entry mode for any model's specs
- Quant picker, KV cache calculator, MoE-aware expert allocation
- Vision support with `--mmproj` and `--no-mmproj-offload`
- OS-specific install instructions (Linux, macOS, Windows, Docker)
- Optional one-liner install script
- Pre-calibrated entropy profiles for curated models

---

## Known-good KV cache path

```text
-ctk q8_0 -ctv turbo3_0 --flash-attn on
```

This is the production default. Other supported cache types parse and run (`f16`, `q8_0`, `q4_0`, `turbo2_0`, `turbo3_0`, `turbo4_0`, `turbo6_0`), but the asymmetric q8 K + turbo3 V path has been most thoroughly validated.

## Current limitations

- The current branch is a stable baseline. Future experimental work should happen on separate branches.
- Path B currently operates at the **layer level** — true per-head granularity is next.
- Calibration requires flash-attn to be temporarily disabled (handled automatically).
- Models with `--reasoning` (thinking) mode need `--reasoning off` for compatible chat API responses.
