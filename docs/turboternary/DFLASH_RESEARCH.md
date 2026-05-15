# DFlash Research: Block Diffusion for Flash Speculative Decoding

**Status:** Research phase — not yet implemented
**Branch:** `dflash-research`
**Date:** 2026-05-15

---

## What Is DFlash?

[DFlash](https://arxiv.org/abs/2602.06036) is a speculative decoding framework that uses a lightweight **block diffusion model** as the drafter instead of an autoregressive one. Key difference from EAGLE-3:

| Aspect | EAGLE-3 | DFlash |
|--------|---------|--------|
| Drafting | Autoregressive (1 token at a time) | **Parallel** (entire block in 1 forward pass) |
| Architecture | 1 transformer layer | 5 transformer layers (full attention) |
| Draft cost | Linear in tokens | **Flat** (same cost for 1 or 16 tokens) |
| Target features | First-layer input only | **KV injection** into every layer |
| Max speedup | 2-3× claimed | **5-6×** on Qwen3-8B |

**Paper claims:** Up to 6.17× lossless acceleration on Qwen3-8B, ~2.5× faster than EAGLE-3.

---

## How DFlash Works

```
┌─────────────────────────────────────────────────────────┐
│                   Target Model (e.g., Qwen3-8B)          │
│  Layer 1  →  Layer 9  →  Layer 17  →  Layer 25  →  33  │
│       │            │           │            │          │  ── feature extraction
│       ▼            ▼           ▼            ▼          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │Fusion 1 │ │Fusion 2 │ │Fusion 3 │ │Fusion 4 │      │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│       │           │           │           │            │
│  ┌────▼───────────▼───────────▼───────────▼──────────┐ │
│  │              DFlash Draft Model (5 layers)         │ │
│  │  Block diffusion → 16 draft tokens in 1 forward   │ │
│  └────────────────────────┬──────────────────────────┘ │
│                           │ draft tokens               │
│                           ▼                            │
│              Target verification (parallel)            │
└─────────────────────────────────────────────────────────┘
```

**Key mechanisms:**

1. **Feature extraction:** After prefill/verification, hidden states from 5 uniformly-sampled target layers are extracted
2. **Feature fusion:** Lightweight projection layers compress multi-layer features
3. **KV injection:** Fused features are injected into *every* draft layer's KV cache — critical difference from EAGLE-3
4. **Block diffusion:** Draft model generates 16 tokens in a single parallel forward pass (single denoising step)
5. **Target verification:** Target model verifies draft tokens in parallel (standard speculative decoding)

The draft model **reuses the target model's embedding and LM head** — only the intermediate 5 layers are trained, keeping params minimal.

---

## Available DFlash Drafters (HuggingFace / z-lab)

### Non-MoE (Dense) Models — Our Primary Interest

| Target Model | DFlash Drafter | Size (bf16) | SGLang | vLLM | Transformers |
|---|---|---|---|---|---|
| **Qwen3-8B** (non-thinking) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) | **2.1 GB** | ✅ | ✅ | ✅ |
| Qwen3-4B (non-thinking) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) | ~1 GB | ✅ | ✅ | ✅ |
| LLaMA-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) | ~2 GB | ✅ | ✅ | ✅ |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) | ~1 GB | ✅ | ✅ | ❌ |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) | ~2 GB | ✅ | ✅ | ❌ |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) | ~2 GB | ✅ | ✅ | ❌ |

### MoE Models (For Reference — DFlash Works but MoE perf in llama.cpp is suboptimal)

| Target Model | DFlash Drafter |
|---|---|
| Qwen3.6-35B-A3B | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| Qwen3.6-27B | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) |
| Gemma-4-31B-it | [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) |
| Gemma-4-26B-A4B-it | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |

### Coming Soon
- DeepSeek-V4-Flash
- DeepSeek-V4-Pro
- GLM-5.1

---

## Support Status in Inference Engines

### 🟢 Production-ready
- **SGLang** — fully supported (--speculative-algorithm DFLASH)
- **vLLM v0.20.1+** — core DFlash support included
- **Transformers** — with trust_remote_code=True
- **MLX** (Apple Silicon) — fully supported

### 🟡 llama.cpp — NOT YET MERGED

**PR #22105** (`ruixiang63`) — "add DFlash support"
- **Status:** Open, waiting on EAGLE-3 PR (#18039) to merge first
- **Based on:** EAGLE-3 encoder-decoder architecture
- **Support:** Qwen3.5/3.6 MoE initially (but perf "currently not optimal")
- **Reason:** MoE + hybrid architecture (Gated DeltaNet + Gated Attention) not well supported by llama.cpp yet
- **Dense model support:** Unknown, but should be easier (no MoE dispatch complexity)
- **Discussion:** [#21569](https://github.com/ggml-org/llama.cpp/discussions/21569)

### 🟢 llama.cpp — MTP (Alternative, Already Works)

**PR #22673** (`am17an`) — "MTP Support"
- **Status:** Active, being tested, merged into upstream `spec-mtp-experiments` branch
- **Performance:** ~1.85× speedup on Qwen3.6-27B (RTX 3090), ~1.3× on Qwen3.6-35B-A3B (6GB mobile)
- **VRAM cost:** ~2.5 GB extra for MTP draft
- **Model needed:** Special MTP GGUF (requires MTP heads in the model)
- **TurboQuant merge reported:** Combined with turbo3 KV, achieves 125 t/s at 98K context on RTX 5060 Ti 16GB

---

## VRAM Analysis for 8GB GPU

### Scenario 1: Qwen3-8B Q4_K_M + DFlash (BEST BET ✅)

| Component | VRAM |
|-----------|------|
| Target model Q4_K_M (~5 GB) | 5,120 MiB |
| DFlash drafter (bf16 2.1 GB → Q4 ~550 MB) | 550 MiB |
| KV cache (8K ctx, q8 K + turbo3 V) | 200 MiB |
| GPU system reservation | 700 MiB |
| **Total** | **~6,570 MiB** ← fits in 8 GB with room |
| Estimated speedup: **5-6×** → from ~60 t/s → ~300-360 t/s |

### Scenario 2: Bonsai 8B Q2_0 + DFlash (No drafter exists ❌)

| Component | VRAM |
|-----------|------|
| Target model Q2_0 (~2.6 GB) | 2,600 MiB |
| DFlash drafter would need training | N/A |
| KV cache (128K ctx) | 2,000 MiB |
| GPU system reservation | 700 MiB |
| **Total** (if drafter existed at Q4 ~550 MB) | **~5,850 MiB** ← would fit |
| But: Bonsai = Qwen3 arch, could use Qwen3-8B-DFlash adapter IF compatible |

### Scenario 3: Ministral-3-3B Q5_K_L + DFlash (No drafter exists ❌)

| Component | VRAM |
|-----------|------|
| Target model Q5_K_L (~2.3 GB) | 2,300 MiB |
| DFlash drafter would need training | N/A |
| KV cache (128K ctx) | 4,100 MiB |
| GPU system reservation | 700 MiB |
| **Total** (if drafter existed) | **~7,100 MiB** ← tight but fits |
| Speedup: could make it much faster for coding, but no drafter available |

**Note re. Bonsai:** Bonsai is Qwen3-based (architecture prefix `qwen3`) with same tokenizer as Qwen3-8B. The Qwen3-8B-DFlash-b16 drafter *might* work with Bonsai since it shares the embedding + LM head architecture. This is worth testing if we can get DFlash running in llama.cpp.

### Scenario 4: Qwen3.6 35B A3B Q5_K_M + MTP (Already possible in llama.cpp ⚠️)

| Component | VRAM |
|-----------|------|
| Target model Q5_K_M (~22 GB) | N/A — CPU MoE offload |
| MTP draft model (~2.5 GB on GPU) | 2,560 MiB |
| GPU system reservation | 700 MiB |
| **Total GPU** | **~3,260 MiB** ← plausible with GPU offloading only some layers |
| But: MTP requires special GGUF with MTP heads | — |

---

## Performance Benchmarks (from paper)

### Qwen3-8B — Greedy (Temp=0)

| Benchmark | Baseline (t/s) | EAGLE-3 (×) | DFlash (×) |
|-----------|:-------------:|:-----------:|:----------:|
| GSM8K | 1× | 2.13× | **5.20×** |
| MATH-500 | 1× | 2.18× | **6.17×** |
| AIME24 | 1× | 2.25× | **5.91×** |
| AIME25 | 1× | 2.18× | **5.85×** |
| HumanEval | 1× | 2.48× | **5.20×** |
| MBPP | 1× | 2.27× | **4.75×** |
| LiveCodeBench | 1× | 2.24× | **5.43×** |
| SWE-Bench | 1× | 1.90× | **2.92×** |
| MT-Bench | 1× | 1.94× | **2.79×** |
| Alpaca | 1× | 1.88× | **2.27×** |

**Key insight:** Coding tasks (HumanEval, MBPP, LiveCodeBench) get 5-6× speedup. Creative writing (MT-Bench, Alpaca) gets ~2.3-2.8×. This aligns with Lyndon's note: "big impact on coding speed, not great on creative writing."

**T=1 sampling results (paper):** DFlash maintains ~4.5× on math, ~2.4× on creative — still strong.

---

## DFlash Drafter Architecture Details

**From Qwen3-8B-DFlash-b16 config.json:**

```json
{
  "architectures": ["DFlashDraftModel"],
  "block_size": 16,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33]
  },
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_hidden_layers": 5,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
  "model_type": "qwen3"
}
```

- **5 layers** (all full attention, no SWA)
- **Architecture:** Standard Qwen3 transformer with custom DFlash feature fusion
- **Reuses:** Target model's embedding, LM head, tokenizer
- **Block size:** 16 draft tokens generated in parallel
- **Feature extraction:** 5 uniformly-sampled target layers
- **Mask token:** 151669 (token placed at positions being drafted)
- **Inference cost:** Single forward pass for all 16 draft tokens (flat cost, doesn't scale with block size)

---

## Paths Forward

### Path A: Qwen3-8B Q4_K_M + DFlash via SGLang/vLLM (Fastest Win 🥇)

This is the quickest path to a working DFlash setup on our hardware:

1. Download `z-lab/Qwen3-8B-DFlash-b16` (~2.1 GB)
2. Download `Qwen/Qwen3-8B` (original bf16, ~16 GB) or GGUF quant
3. Run via **SGLang** or **vLLM** with DFlash support

**vLLM command:**
```bash
vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-8B-DFlash-b16", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

**But wait:** The target model needs to be loaded in bf16 (vLLM/SGLang don't support GGUF). On 8GB GPU, bf16 Qwen3-8B (~16 GB) is impossible.

**GGUF alternative:** If we could get DFlash working in **llama.cpp**, we could use the Q4_K_M quant (~5 GB). This is Path B.

### Path B: DFlash in llama.cpp (Longer — Need to Contribute to PR #22105)

1. Wait for EAGLE-3 PR (#18039) to merge → then DFlash PR (#22105) should be rebased
2. OR: Fork PR #22105, apply to our TurboQuant fork
3. Modify to support dense models (Qwen3-8B)
4. Add GGUF conversion for DFlash drafters (convert_hf_to_gguf.py changes needed)
5. Test with Qwen3-8B Q4_K_M + DFlash drafter converted to GGUF

**llama.cpp files that will need changes:**
- `src/models/` — new DFlash architecture (similar pattern to eagle3.cpp)
- `common/speculative.h/.cpp` — extend `common_speculative_type` with DFLASH
- `llama.h` — new `LLM_ARCH_DFLASH` enum
- Feature extraction hooks from target model layers
- `convert_hf_to_gguf.py` — DFlash HF→GGUF conversion
- `--dflash` flag in CLI/server

### Path C: Alternative — MTP (Already Works in llama.cpp)

If DFlash is too far from production in llama.cpp, MTP is already working:

1. Get MTP GGUF of a dense model (there's Qwen3.6-27B-MTP-Q6_K available)
2. Use `--spec-type mtp --spec-draft-n-max 3` flags
3. VRAM cost: ~2.5 GB extra (unlikely to fit with Qwen3.6 35B A3B on 8GB)

**Better MTP target for 8GB:** Qwen3-8B-Q4_K_M (~5 GB) + MTP (~1 GB for smaller draft)? Would need a Qwen3-8B MTP GGUF to be created.

### Path D: Bonsai 8B + Qwen3-8B-DFlash (Research — Compatibility Unknown)

Since **Bonsai** is based on **Qwen3** architecture with the same tokenizer:
- The DFlash drafter reuses the target's embedding + LM head
- Bonsai has the same vocab_size (151936), same tokenizer
- The drafter extracts features from target layers 1,9,17,25,33 — Bonsai has 36 layers, same structure
- **Hypothesis:** The Qwen3-8B-DFlash-b16 drafter could work with Bonsai 8B Q2_0 out of the box
- **Unknown:** Whether Bonsai's Q2 quantization and different hidden structure would break feature extraction

---

## VRAM-Fitting Strategy with Our Tricks

The real power is combining our existing research with DFlash:

```
DFlash speedup (5-6×)
  × Entropy Path B KV savings (~22%)
  × turbo3 V cache (2.5× KV compression)
  × Q4 quant of target model
  = Massive net speedup for dense coding models on 8GB
```

**What fits in 8GB with DFlash:**

| Model | Quant | Base Size | + DFlash Q4 | NV Cache (32K) | Total | Fits? |
|-------|-------|-----------|-------------|----------------|-------|-------|
| Qwen3-8B | Q4_K_M | ~5.0 GB | ~0.55 GB | ~0.8 GB | **~6.3 GB** | ✅ Yes |
| Bonsai 8B | Q2_0 | ~2.6 GB | ~0.55 GB | ~2.0 GB | **~5.2 GB** | ✅ Yes |
| Ministral-3-3B | Q5_K_L | ~2.3 GB | (none) | ~4.1 GB | N/A | ❌ No drafter |

---

## Recommended Research Path

### Phase 1: Test DFlash via SGLang/vLLM (if VRAM permits)
- Download Qwen3-8B-DFlash-b16 (~2.1 GB)
- Test with `transformers` backend using CPU offload for target
- Measure actual speedup vs baseline

### Phase 2: llama.cpp Integration
- Track PR #22105 and PR #18039
- Branch from upstream's `spec-mtp-experiments` (has latest spec infrastructure)
- Add DFlash as a new `common_speculative_type`
- Implement GGUF conversion for DFlash drafters
- Add feature extraction hooks to target model

### Phase 3: Test with Our Models
- Qwen3-8B Q4_K_M as primary target
- Bonsai 8B Q2_0 as compatibility test (might work with same drafter)
- Benchmark with CodeNeedle

### Phase 4: Combine with Entropy + TurboQuant
- DFlash + Entropy Path B + turbo3 V cache
- Measure end-to-end speed at various context sizes
- Add DFlash support to MostlysaneAI config generator

---

## References

- **Paper:** https://arxiv.org/abs/2602.06036
- **GitHub:** https://github.com/z-lab/dflash
- **Blog:** https://z-lab.ai/projects/dflash/
- **HF Collection:** https://huggingface.co/collections/z-lab/dflash
- **llama.cpp DFlash PR:** https://github.com/ggml-org/llama.cpp/pull/22105
- **llama.cpp MTP PR:** https://github.com/ggml-org/llama.cpp/pull/22673
- **llama.cpp DFlash Discussion:** https://github.com/ggml-org/llama.cpp/discussions/21569
