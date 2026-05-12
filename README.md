# llama.cpp — Ternary 1.58-bit + TurboQuant fork

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Upstream](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-blue)](https://github.com/ggml-org/llama.cpp)

**512K token context on a single 8 GB GPU — at full quality.**

This fork extends `llama.cpp` with **entropy-guided KV cache compression**, achieving ~30% memory savings with zero quality loss. The highlight: Qwen3.6 35B A3B (a 35B MoE model) runs with 512K context on an RTX 3070 Ti.

## Quick start

```bash
# Build (CUDA)
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run Qwen3.6 with entropy-guided KV cache — 256K context
llama-server -m Qwen3.6-35B-A3B-Q5_K_M.gguf \
  -ngl 99 --n-cpu-moe 39 --ctx-size 256000 --flash-attn on \
  -ctk q8_0 -ctv turbo3_0 \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0
```

> **512K context?** Use `--n-cpu-moe 40` instead of 39. See the [full docs](docs/turboternary/README.md) for details.

## Key features

- **Entropy-adaptive KV cache** — per-layer mixed precision (q8 K for attention layers, turbo4 K for the rest)
- **TurboQuant types** — turbo2/3/4/6 for long-context inference
- **Ternary Bonsai / Q2_0 model support** — 1.58-bit GGUF models
- **MoE CPU expert offloading** — run 35B models on 8 GB VRAM
- **512K context validated** — real-world testing with a 35B model on RTX 3070 Ti

## 🎨 Quality showcase

### Walking dog animation — single-file HTML

A popular single-file HTML animation test seen on social media. Qwen3.6 35B A3B with entropy-guided KV cache (Path B, ratio 2.0) running on a single 8 GB GPU produced a visually rich walking scene (person + articulated dog + leash + parallax background) on the **first attempt**, resolving the single animation-loop bug on a clean re-prompt.

Download and open any of these in your browser:

| Model | File | Result | Time |
|-------|------|--------|------|
| **Qwen3.6 35B A3B** ⭐ | [`qwen3.6_walking_dog_animation_test.html`](demos/qwen3.6_walking_dog_animation_test.html) | ✅ Fully working — articulated figures, correct limb motion, dog walking forwards, leash physics, parallax background | 5 min 23 s (10,866 tok @ 33.6 t/s) |
| ChatGPT 5.5 | [`ChatGPT5-5_walking_dog_animation_test.html`](demos/ChatGPT5-5_walking_dog_animation_test.html) | ❌ No animation, layout issues | 3 min |
| Claude Sonnet 4.6 | [`Claude_Sonnet_4-6_walking_dog_animation_test.html`](demos/Claude_Sonnet_4-6_walking_dog_animation_test.html) | ⚠️ Good scene, but person's arms swing incorrectly and dog walks backwards | 4 min |
| Gemini Pro 3.1 | [`Gemini_pro_3-1_walking_dog_animation_test.html`](demos/Gemini_pro_3-1_walking_dog_animation_test.html) | ⚠️ Animates fairly well, but not as polished | 5 min |

> Qwen3.6 was the only model running locally on consumer hardware (Ryzen 5 3800X, 32 GB DDR4-3600, RTX 3070 Ti 8 GB). The cloud models had their full server-side compute available. Despite this, Qwen3.6 produced the best result — the only fully working, correctly animated output with proper limb articulation and dog orientation.

Hardware: Ryzen 5 3800X, 32 GB DDR4-3600, RTX 3070 Ti 8 GB. Generated 10,866 tokens in 5 min 23 sec (avg 33.6 t/s), with 11.4 sec reasoning time.

Full prompt and comparison details in [`benchmark-results/quality-prompts.md`](benchmark-results/quality-prompts.md).

The Qwen walking dog demo will also be featured as a live preview demo on the **Mostlysane Local AI** web app (see remaining research tasks).

## 📖 Full documentation

All validation results, calibration instructions, recommended configs, and architecture notes are in **[`docs/turboternary/README.md`](docs/turboternary/README.md)**.

---

The upstream `llama.cpp` README follows below. The standard build, server, API, and model documentation still applies unless noted otherwise.

## What this branch adds

### Ternary Bonsai / Q2_0 support

This branch carries Prism-derived Q2_0 / ternary model support and has been validated with:

- `Ternary-Bonsai-8B-Q2_0.gguf`
- CUDA execution on an NVIDIA RTX 3070 Ti
- `llama-cli`, `llama-server`, and `llama-bench` workflows

The Ternary Bonsai model has been useful as a fast local helper and as a quality canary for TurboKV experiments because it is small enough to test quickly while still being sensitive to KV-cache retrieval errors.

### TurboQuant / TurboKV integration

Added ggml KV-cache types:

- `turbo2_0`
- `turbo3_0`
- `turbo4_0`
- `turbo6_0` (6-bit PolarQuant K from pitcany/ollama-turboquant port)

#### KV-cache recommendations (settled from extensive testing)

| Config | Quality | Memory vs q8/q8 | Use case |
|--------|---------|----------------|----------|
| `-ctk q8_0 -ctv turbo3_0` | ✅ Best | ~530 MiB saved | Production default — best quality/memory balance |
| `-ctk turbo6_0 -ctv turbo3_0` | ⚠️ Minor generative trade-off | ~1100 MiB saved | Memory-constrained scenarios — fits larger context at slight quality cost |
| `-ctk turbo4_0 -ctv turbo3_0` | ❌ Significant regression | ~1400 MiB saved | Not recommended — fails code retrieval (3/20 send_head) and adds hallucinations |

**Why the split?** Testing showed compressed V (turbo3) is universally useful, but aggressive K compression degrades quality differently across task types:
- Retrieval (CodeNeedle): turbo6 K passes 20/20 send_head; turbo4 K fails at 3/20
- Generative coding (Prompt-Vault): turbo6 K needs 1-2 more retries than q8 K for equivalent results

For Bonsai CodeNeedle, `K=q8_0/V=turbo3_0` kept quality high while still saving memory versus q8/q8.

### CUDA and Metal integration

The branch includes:

- TurboKV cache construction and CLI/server parsing for Turbo cache types.
- CUDA FlashAttention support for mixed TurboKV combinations.
- CUDA dequantize-to-f16 FlashAttention fallback coverage for compatible mixed TurboKV paths.
- F16-K + Turbo-V CUDA FlashAttention dispatch.
- Metal TurboKV support merged into the stable branch.
- Experimental env-gated adaptive K-cache knobs for controlled testing:

```bash
TURBO_K_Q8_FIRST_N=<layers>
TURBO_K_Q8_LAST_N=<layers>
```

Those adaptive knobs are deliberately experimental. They are useful for probing layer sensitivity, but Bonsai `send_head` testing showed partial adaptive K did **not** rescue the failing full-compressed-K case; only all-q8 K passed that target.

## Recommended commands

### CUDA build used locally

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build --target llama-cli llama-server llama-bench -j8
```

Older local builds also used static libstdc++/libgcc linker flags to avoid local CUDA/libstdc++ mismatches. If your toolchain is clean, the command above is preferred.

### Ternary Bonsai server — entropy-guided Path B (recommended)

Book-calibrated entropy profile at full 65K context. Uses Path B per-layer mixed precision.

```bash
./build/bin/llama-server \
  -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf \
  --alias ternary-bonsai-q2 \
  -ngl 99 \
  --ctx-size 65536 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --entropy-profile entropy_profile_bonsai_book.json \
  --entropy-prune-ratio 2.0
```

### Ternary Bonsai server — memory-saving (turbo6 low-K + entropy)

Uses turbo6 K for low-entropy layers via entropy-guided Path B. Saves additional VRAM while protecting retrieval-sensitive layers.

```bash
./build/bin/llama-server \
  -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf \
  --alias ternary-bonsai-q2 \
  -ngl 99 \
  --ctx-size 65536 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --entropy-profile entropy_profile_bonsai_book.json \
  --entropy-prune-ratio 2.0 \
  --entropy-low-k-type turbo6
```

### Qwen3.6 35B A3B MoE — entropy-guided Path B (recommended)

256K context with book-calibrated entropy profile. The production default — 22% KV memory savings with no quality loss (11/11 CodeNeedle, 15/16 jQuery).

```bash
./build/bin/llama-server \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  --alias qwen3.6-35b-a3b \
  -ngl 99 \
  --n-cpu-moe 39 \
  --ctx-size 256000 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0
```

> **512K context?** Use `--n-cpu-moe 40` instead of 39.

### Qwen3.6 35B A3B MoE — turbo6 low-K memory-saving

Uses turbo6 K for low-entropy attention layers. Saves ~26% KV memory with only minor generative trade-off. Verified passes all CodeNeedle retrieval gates.

```bash
./build/bin/llama-server \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  --alias qwen3.6-35b-a3b \
  -ngl 99 \
  --n-cpu-moe 39 \
  --ctx-size 256000 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0 \
  --entropy-low-k-type turbo6
```

### Qwen3-VL 30B A3B — entropy-guided Path B

Vision MoE with book-calibrated entropy profile. Note: requires `--mmproj` for vision tasks.

```bash
./build/bin/llama-server \
  -m ~/AI/models/Qwen_Qwen3-VL-30B-A3B-Instruct-Q5_K_L.gguf \
  --mmproj ~/AI/models/mmproj-Qwen_Qwen3-VL-30B-A3B-Instruct-f16.gguf \
  --alias qwen3-vl-30b-a3b \
  -ngl 99 \
  --n-cpu-moe 39 \
  --no-mmproj-offload \
  --ctx-size 128000 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --entropy-profile entropy_profile_qwen3vl_book.json \
  --entropy-prune-ratio 2.0
```

### Qwen3.6 35B A3B — CLI for batch/generation tasks

```bash
./build/bin/llama-cli \
  -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 35 \
  --ctx-size 200000 \
  --flash-attn on \
  -ctk q8_0 \
  -ctv turbo3_0 \
  --reasoning off \
  --entropy-profile entropy_profile_qwen_book.json \
  --entropy-prune-ratio 2.0
```

For Qwen reasoning models, `--reasoning off` is recommended for benchmark harnesses such as CodeNeedle unless thinking output is explicitly desired.

### Self-calibration (creating your own entropy profile)

Run the calibration tool with a long-form text file to generate a model-specific entropy profile:

```bash
./build/bin/llama-entropy-calibrate \
  -m ~/AI/models/Your-Model.gguf \
  --cal-text /path/to/reference_text.txt \
  --flash-attn on -ngl 99 \
  -ctk f16 -ctv f16 -c 512
# Produces entropy_profile.json — pass to --entropy-profile
```

> Calibration requires flash-attn to be temporarily disabled internally. Use f16 KV types and a context size that fits your GPU's compute buffer budget. For large models (>15B), reduce `-ngl` to free VRAM.

## Testing and benchmark summary

Detailed logs live in:

- [`docs/turboternary/`](docs/turboternary/)
- [`benchmark-results/2026-05-07-turboternary/`](benchmark-results/2026-05-07-turboternary/)

### Ternary Bonsai performance

Validated on `Ternary-Bonsai-8B-Q2_0.gguf` with various TurboKV configs (all `--flash-attn on`):

**Benchmark (`llama-bench -p 512 -n 128 -r 1`):**
- `q8_0/V=turbo3_0`: ~3372 pp512 t/s, ~107 tg128 t/s
- `turbo4_0/V=turbo3_0`: ~3268 pp512 t/s, ~104 tg128 t/s
- `turbo6_0/V=turbo3_0`: ~3168 pp512 t/s, ~99 tg128 t/s

**Memory at 32k context (RTX 3070 Ti 8GB):**
| Config | KV MiB | K MiB | V MiB |
|--------|--------|-------|-------|
| q8_0 + turbo3_0 | 1918 | 1468 | 450 |
| turbo6_0 + turbo3_0 | 1350 | 900 | 450 |
| turbo4_0 + turbo3_0 | 1062 | 612 | 450 |

**Long-context memory smoke tests (RTX 3070 Ti):**
- turbo6 at 65k (full Bonsai context): `2700 MiB` KV, ~2116 MiB free VRAM — fits comfortably
- 96k, `K=turbo4_0/V=turbo3_0`: fit fully in GPU RAM, KV `3186 MiB`
- 128k, `K=turbo4_0/V=turbo3_0`: allocation and coherent short generation passed, KV `4248 MiB`
- 128k, `K=q8_0/V=turbo3_0`: allocation passed, KV `6696 MiB`

### Qwen3.6 35B A3B performance

Validated with `Qwen3.6-35B-A3B-Q5_K_M.gguf`, CPU MoE offload, 256k context setting, and TurboKV:

- Short 256k-context run: prompt about `109 tok/s`, generation about `45 tok/s`.
- Longer local run held around `43 tok/s`, matching or slightly exceeding the older dedicated TurboQuant fork in this environment.
- After the CUDA prefill fix, `llama-bench` smoke results with CPU MoE experts were around `pp512 443 tok/s`, `tg32 42 tok/s`; `pp4096 429 tok/s`, `tg16 42 tok/s`.

### CodeNeedle quality testing

CodeNeedle has been used as a retrieval/code-quality gate, not just a speed benchmark:

**Qwen3.6 35B `http_server` (32k context):**
| Config | Score | Primary | Halluc | Bonus |
|--------|-------|---------|--------|-------|
| No compression | 11/11 | 220/220 | 0 | 0 |
| q8_0 + turbo3_0 | 11/11 | 220/220 | 0 | 0 |
| turbo4_0 + turbo3_0 | 11/11 | 214/220 | 23 | 81 |

**Qwen3.6 jQuery comparison:**
| Config | Score | Primary | Halluc | Bonus |
|--------|-------|---------|--------|-------|
| No compression | 15/16 | 273/320 | 31 | 15 |
| turbo4_0 + turbo3_0 | 15/16 | 276/320 | 63 | 81 |
| High-quality K + compressed V | 15/16 | 287/320 | 32 | 43 |

**Bonsai 8B `http_server` 32k — the critical `send_head` comparison:**
| Config | Score | Primary | Halluc | Bonus | send_head |
|--------|-------|---------|--------|-------|-----------|
| q8_0 + q8_0 | 11/11 | 207/220 | 218 | 223 | ✅ 20/20 |
| q8_0 + turbo3_0 | 11/11 | 218/220 | 194 | 231 | ✅ 20/20 |
| **turbo6_0 + turbo3_0** | **10/11** | **203/220** | **219** | **207** | **✅ 20/20 (standalone)** |
| turbo4_0 + turbo3_0 | 10/11 | 181/220 | 885 | 199 | ❌ 1/20 |

Turbo6 passes the `send_head` gate at 20/20 when tested standalone — only turbo4 fails catastrophically. The sequential full-suite 10/11 result for turbo6 is due to function-name ambiguity in the corpus (two `send_head` variants in the same file), not K quality.

Interpretation: raw speed and recall are not enough. KV compression must also preserve useful coding/assistant behavior.

### Prompt-Vault real-world build testing

Prompt-Vault has been used as an additional practical coding/assistant benchmark: <https://github.com/w512/Prompt-Vault>. These tests ask the model to build small browser apps across easy (ToDo, Bubble Sort), medium (Sorting Visualization), and hard (Kanban Board) prompts. This catches failures that retrieval or short coherence tests miss.

**Qwen3.6 35B Q5_K_M at 200k context (~36-40 tok/s):**

| Level | q8 K + turbo3 V | turbo6 K + turbo3 V |
|-------|----------------|--------------------|
| ToDo List | ✅ Pass — fully functional, good looking | ✅ Pass — fully functional, even without reasoning |
| Bubble Sort | ✅ Pass — fully functional, looks great | ✅ Pass — first attempt lacked Reset, clean second attempt |
| Sorting Viz | ✅ Pass — fully functional, looks great | ✅ Pass — visual issues first pass, resolved with retry |
| Kanban Board | ✅ Pass — partial first pass, fully working second pass | ✅ Pass — functional but less tidy, lacks side-scroll |

**Bonsai 8B Q2_0 at 65k context (~70-90 tok/s):**

| Level | q8 K + turbo3 V | turbo6 K + turbo3 V |
|-------|----------------|--------------------|
| ToDo List | ❌ Partial — mark-item-done never worked | ✅ Pass — simpler but functional |
| Bubble Sort | ❌ Fail — UI only after 3 attempts | ❌ Fail — same outcome |
| Sorting Viz | ❌ Partial — UI only, no data | ❌ Partial — same as q8 |
| Kanban Board | — Skipped (beyond capability) | — Skipped |

**Key takeaway:** Turbo6 K shows a subtle generative quality trade-off. Qwen3.6 remains a solid coding partner with turbo6, but may need 1-2 extra retries vs q8 K for equivalent results. Bonsai is already at its capability ceiling; turbo6 doesn't make it worse but doesn't help either.

For production coding, `K=q8_0/V=turbo3_0` remains the recommended path. Use `turbo6 K` when memory is the binding constraint and you're willing to accept slightly less consistent first-pass outputs.

## Current technical notes

### Turbo3 block size

This branch already uses Turbo3 block size 128:

```text
QK_TURBO3       = 128
QK_TURBO3_GROUP = 128
block_turbo3_0  = 50 bytes
3.125 bits/value, about 5.12x versus fp16 KV
```

The CUDA set-rows path already launches 128-thread WHT groups for this mode.

### Recommended KV-cache guidance (settled)

Through extensive CodeNeedle retrieval testing and Prompt-Vault generative coding tests across Qwen3.6 and Bonsai models, the following recommendations are established:

- **Production default:** `-ctk q8_0 -ctv turbo3_0` — best quality, best generative consistency, ~530 MiB savings vs q8/q8
- **Memory-constrained (recommended fallback):** `-ctk turbo6_0 -ctv turbo3_0` — passes all retrieval gates, minor generative trade-off, ~1100 MiB savings vs q8/q8. Use when you need to fit a larger context on limited VRAM.
- **Avoid:** `-ctk turbo4_0` — fails code retrieval benchmarks (3/20 send_head) and adds hallucinations even on large models
- **Known risk:** Full compressed K can subtly degrade generative coding consistency even when retrieval looks fine. Test both if quality is critical.
- **Not promising (discounted):**
  - Layer-only adaptive K protection (env knobs `TURBO_K_Q8_FIRST_N` / `TURBO_K_Q8_LAST_N`) — Bonsai `send_head` failed unless all K layers were q8; any single turbo-compressed layer could break retrieval.
  - Pre-RoPE K quantization (branch `turboternary-poc-prerope-k`) — numerical validation showed +0.02 dB SNR improvement for turbo4 pre-RoPE vs post-RoPE, i.e. essentially zero. The quality regression with 4-bit K is inherent to the precision, not the RoPE ordering. See `tools/prerope-validate.cpp`.

## Research sources and references

This branch pulls ideas, comparisons, or implementation clues from several places. Licensing matters: not all code is suitable for direct copying.

- Upstream `llama.cpp`: <https://github.com/ggml-org/llama.cpp>
- This fork: <https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>
- PrismML llama.cpp work: <https://github.com/PrismML-Eng/llama.cpp>
- TheTom llama.cpp TurboQuant work: <https://github.com/TheTom/llama-cpp-turboquant>
- TheTom TurboQuant Plus: <https://github.com/TheTom/turboquant_plus>
- 0xSero TurboQuant: <https://github.com/0xSero/turboquant> — GPL-3.0 local repo; use as architectural reference only, do not copy into this MIT fork.
- tonbistudio TurboQuant PyTorch: <https://github.com/tonbistudio/turboquant-pytorch> — useful corrected findings around MSE-only/no-QJL, asymmetric K/V, residual windows, and generation quality.
- pitcany Ollama TurboQuant: <https://github.com/pitcany/ollama-turboquant> — future runtime/kernel integration lead.
- CodeNeedle benchmark: <https://github.com/alexziskind1/codeneedle>
- Prompt-Vault prompt suite: <https://github.com/w512/Prompt-Vault>
- Entropy-adaptive KV-cache lead: <https://github.com/SCJedi/entropy-adaptive-kv-cache>
- Independent TurboQuant landscape analysis: <https://www.frr.dev/posts/turboquant-one-month-later-implementations-controversy-benchmarks/>

Key research conclusions driving this branch:

- MSE/PolarQuant-style compression is useful; QJL has not looked attractive in practical community implementations.
- Asymmetric K/V precision is important: keys need more care than values.
- Code and assistant-style tests are necessary; MSE/cosine/needle metrics alone can miss quality failures.
- Turbo6 (6-bit PolarQuant) is a viable memory-saving K format that passes all retrieval gates with only minor generative trade-off.
- Turbo4 (4-bit PolarQuant) is not recommended — it fails code retrieval and adds hallucinations.
- For real K memory savings beyond turbo6, the next path would likely be a higher-precision MSE-only K format or residual-window approaches (keep last N tokens at full precision, compress the rest).

## Branch policy

Treat the current branch as a working integration branch. More invasive work should happen on feature branches, for example:

- `turboternary-turbo-k5-k6`
- `turboternary-residual-window`
- `turboternary-ollama-runtime-review`
- `turboternary-prompt-vault-eval`

## Upstream llama.cpp README

The remaining sections are inherited from upstream `llama.cpp`.

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
