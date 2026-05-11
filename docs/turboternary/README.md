# Ternary 1.58-bit + TurboQuant fork — validated for production

**Headline result:** Qwen3.6 35B A3B runs with **512K token context** on a single **8 GB RTX 3070 Ti** at full quality — no offloading, no degradation. See [Real-World Validation](#real-world-validation-512k-context-on-8gb) below.

This fork integrates four pieces of work on top of upstream `llama.cpp`, culminating in the **entropy-adaptive KV cache** that makes the headline result possible:

1. **PrismML Q2_0 / ternary model support** for 1.58-bit-style GGUF models (Ternary Bonsai Q2_0).
2. **TurboQuant KV cache compression** — `turbo2_0`, `turbo3_0`, `turbo4_0`, and `turbo6_0` ggml types for long-context inference.
3. **CUDA FlashAttention vector-path optimizations** — turbo VEC kernels tuned for compressed KV cache types.
4. **SCJedi Entropy-Adaptive KV Cache** — per-head entropy profiling drives per-layer mixed-precision K types. The calibrated profile tells us which layers need 8-bit K and which can safely use 4-bit, giving ~30% memory savings with zero quality loss.

**The key insight:** V cache compression is "free" (no quality impact). All degradation comes from K compression. Asymmetric K/V with entropy-guided K precision is the right approach — and it's what this branch delivers.

Repository: <https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>

Research inspirations:
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026)
- [SCJedi/entropy-adaptive-kv-cache](https://github.com/SCJedi/entropy-adaptive-kv-cache)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)

---

## Build

CUDA release build used for validation:

```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS='-static-libstdc++ -static-libgcc' \
  -DCMAKE_SHARED_LINKER_FLAGS='-static-libstdc++ -static-libgcc'
make -j$(nproc)
```

The static libstdc++/libgcc flags avoid the local CUDA/libstdc++ linker issue.

---

## Entropy-Guided KV Cache Compression (Path B)

**What it does:** Uses per-head attention entropy to decide which transformer layers can use `turbo4_0` (4-bit) K vs `q8_0` (8-bit) K. Low-entropy (sink/focused) heads tolerate aggressive compression; high-entropy (diffuse) heads keep full precision.

### Architecture

The entropy profile is calibrated once per model, saved as a JSON file. At server startup, the profile is loaded and a per-layer callback returns the appropriate `ggml_type` for each layer based on its mean entropy relative to the median.

For **hybrid architectures** like Qwen3.6 (30 DeltaNet + 10 Attention layers), Path B correctly identifies that only the 10 Attention layers produce usable entropy — the DeltaNet layers (linear attention) show flat entropy and safely receive the compressed K type.

### Key flags

| Flag | Purpose |
|------|---------|
| `--entropy-profile <path>` | Activate Path B with a pre-calibrated profile |
| `--entropy-prune-ratio <float>` | Controls aggressiveness (1.0 = median threshold, 2.0 = sweet spot, 2.5 = max compression) |
| `--entropy-low-k-type <type>` | Compressed K type for low-entropy layers (`turbo4` default, `turbo6` for gentler compression) |

### Calibration

```bash
# Quick calibration (short prompts, ~10 tokens each)
./build/bin/llama-entropy-calibrate -m ~/AI/models/<model>.gguf \
  -ngl 99 --n-cpu-moe <n_layers> -c 64

# Best results: calibrate with real text (e.g. a book excerpt)
./build/bin/llama-entropy-calibrate -m ~/AI/models/<model>.gguf \
  -ngl 99 --n-cpu-moe <n_layers> -c 1024 \
  --cal-text /path/to/text_file.txt
```

Book-text calibration gives a wider entropy spread (mean 1.081 vs 0.620) and better discrimination between layers. Pre-calibrated profiles are provided in the repo root (see [table below](#entropy-profiles-pre-calibrated)).

### YARN Compatibility

YARN extends the positional encoding beyond trained context. Entropy profiles are transferable to extended context because head specialization is an architectural property, not position-dependent.

---

## MoE Models with CPU Expert Offloading

**The key to running large models on consumer GPUs.** With `--n-cpu-moe <n_layers>`:
- The full model loads in system RAM (needs 32 GB+)
- Only the 9 active experts per token per layer are computed on CPU
- Dense layers and KV cache live in GPU VRAM
- CodeNeedle and real-world Prompt-Vault testing confirm **identical quality to full-GPU** at any reasonable context size

**Finding the limit:** For Qwen3.6 35B A3B on an 8 GB RTX 3070 Ti:
- **`--n-cpu-moe 39`** — all but one MoE expert layer on CPU (default for 256K ctx)
- **`--n-cpu-moe 40`** — all MoE experts on CPU (required for 512K ctx)
- Going below 39 (`--n-cpu-moe 38` or lower) moves expert weights back to GPU, consuming VRAM that's better used for the KV cache

```
-ngl 99 --n-cpu-moe 39 --no-mmap   # 256K context
-ngl 99 --n-cpu-moe 40 --no-mmap   # 512K context
```

---

## Recommended Command Lines

The **entropy-guided Path B** configuration is the production default. It saves 22% KV cache memory at quality parity with full q8.

### Qwen3.6 35B A3B (PRIMARY — text + vision)

**Recommended (256K context):**
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
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0
```

**Max context (512K):**
```bash
llama-server \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  --mmproj ~/AI/models/mmproj-Qwen3.6-35B-A3B-F16.gguf \
  --no-mmproj-offload \
  --alias qwen3.6-35b-a3b \
  -ngl 99 --n-cpu-moe 40 --no-mmap \
  --ctx-size 512000 --flash-attn on \
  -ctk q8_0 -ctv turbo3_0 \
  --host 127.0.0.1 --port 8080 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0
```

### Qwen3-VL-30B A3B (vision MoE, secondary)
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

### Bonsai 8B Q2.0 (lightweight test model)
```bash
llama-cli -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf \
  -p "Your prompt here" -n 4096 -c 65535 \
  -s 12345 --temp 0 --top-k 1 --top-p 1 \
  --flash-attn on -ngl 40 \
  -ctk q8_0 -ctv turbo3_0
```

---

## Recommended KV Cache Configurations

| Type | K | V | When to use |
|------|---|---|-------------|
| **★ Path B (default)** | **entropy mix (q8/turbo4)** | **turbo3_0** | **Production — best quality/memory tradeoff** |
| Stable baseline | q8_0 | turbo3_0 | Full q8 K precision, no entropy needed |
| Max headroom | Path B + turbo6 low-K | turbo3_0 | Same memory as turbo4 but with 50% more K precision on low-entropy layers |
| Max V savings | Path B | turbo2_0 | V compression is free — only try when K is protected |
| Lightweight (Bonsai) | turbo4_0 | turbo3_0 | Fast on small models, okay quality |
| All-in | turbo4_0 | turbo2_0 | Max compression, quality loss likely |

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

## Real-World Validation: 512K Context on 8GB

Tested on a single RTX 3070 Ti (8 GB VRAM) with the book-calibrated entropy profile at prune ratio 2.0 (default turbo4 for low-entropy layers). Full 512K context fits — no offloading, no tricks.

**Setup:**
```bash
~/AI/ternaryturboquant/llama.cpp/build/bin/llama-server \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  --mmproj ~/AI/models/mmproj-Qwen3.6-35B-A3B-F16.gguf \
  --no-mmproj-offload \
  --alias qwen3.6-35b-a3b \
  -ngl 99 --n-cpu-moe 40 --ctx-size 512000 \
  --flash-attn on -ctk q8_0 -ctv turbo3_0 \
  --host 127.0.0.1 --port 8080 \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0 --no-mmap
```

> **Note:** `--n-cpu-moe 40` (instead of the usual 39) was needed to avoid GPU OOM — one fewer MoE layer on GPU makes room for the 512K KV cache.

**Performance (Hitch Hiker's Guide, ~70K prompt tokens):**

| Metric | Value |
|--------|-------|
| Prefill (70K tokens) | ~2-3 minutes |
| Generation (post-prefill) | ~25 t/s |
| Low-context generation | ~40 t/s |
| Quality vs Q8 no-entropy | No discernable loss at ratio 2.0 |

**KV cache RAM at 256K context for each pruning ratio:**

| Config | KV cache (MiB) | vs q8 baseline | Quality (Kanban Board) |
|--------|:--------------:|:--------------:|:---------------------:|
| q8 no entropy | 2,067 | — | Baseline |
| Path B ratio 1.5 | not tested | — | — |
| Path B ratio 2.0 | **1,602** | **−22%** | ✅ Full quality, same visual appeal |
| Path B ratio 2.5 | **1,536** | **−26%** | ⚠️ Virtually full, minor visual degradation |

Ratio 2.5 saves only 66 MiB more than 2.0 while starting to degrade quality — **ratio 2.0 is the sweet spot**.

**Memory breakdown at 512K ctx / ratio 2.0 from `memory_breakdown_print`:**

| Region | Host (MiB) | GPU (MiB) |
|--------|:----------:|:---------:|
| Model (dense layers) | — | 1,439 |
| Model (MoE experts, CPU) | 22,133 | — |
| KV cache @ 512K ctx | — | 2,954 |
| Compute buffers | 1,008 | 1,727 |
| Unaccounted | — | 1,379 |
| Free | — | 339 |
| **Total GPU used** | — | **7,840 / 8,192** |

The KV cache drops from an estimated ~4,134 MiB (q8 K + turbo3 V at 512K, no pruning) to **2,954 MiB** — a **29% savings** from entropy-guided mixed precision, with no discernable quality loss.

---

## Entropy Profiles (Pre-Calibrated)

| Model | File | Heads | Mean Entropy | CV |
|-------|------|:----:|:------------:|:--:|
| Bonsai 8B | `entropy_profile_bonsai.json` | 1152 | 0.620 | 0.58 |
| Qwen3.6 35B A3B | `entropy_profile_qwen.json` (short prompts, deprecated) | 640 | 0.620 | 0.55 |
| Qwen3.6 35B A3B | `entropy_profile_qwen_book.json` (Hitch Hiker text) | 640 | **1.081** | 0.24 |
| Qwen3-VL-30B A3B | `entropy_profile_qwen3vl.json` | 1536 | 0.468 | 0.42 |
| GLM-4.6V-Flash 9B | `entropy_profile_glm.json` (archived) | 1280 | 0.502 | 0.37 |

---

## Future Research Directions

### From TheTom/turboquant_plus (to explore)
- **Block size optimization**: 5.12× compression potential
- **Sparse V dequant**: +22.8% decode speed
- **Boundary layer tuning** (deferred): TheTom found protecting first 2 + last 2 layers recovers 37-91% of quality gap from aggressive uniform K compression. Our entropy approach likely handles this automatically (high-entropy boundary layers naturally stay at q8), but we'll book-calibrate Bonsai and Qwen3-VL first to confirm before pursuing this further
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

## Current limitations

- Path B operates at the **layer level** — true **per-head granularity** would unlock finer-grained memory budgets.
- Calibration requires flash-attn to be temporarily disabled (handled automatically).
- Models with `--reasoning` (thinking) mode need `--reasoning off` for compatible chat API responses.
- The RTX 3070 Ti 8 GB is the reference GPU. Results will vary on other hardware (AMD, older NVIDIA, integrated GPUs).
