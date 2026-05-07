# Benchmark results — Ternary + TurboQuant branch — 2026-05-07

## Environment

- Repo: `~/AI/ternaryturboquant/llama.cpp`
- Branch: `turboternary`
- Code checkpoint before docs: `5c136e3ba` (`Allow Turbo KV cache types in llama-bench`)
- GPU: NVIDIA GeForce RTX 3070 Ti, 7840 MiB VRAM
- Build: CUDA Release with static libstdc++/libgcc linker flags

Raw logs are stored under:

```text
benchmark-results/2026-05-07-turboternary/
```

## Bonsai 8B Q2_0 KV matrix

Command shape:

```bash
./bin/llama-bench -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf -ngl 99 -fa 1 -ctk <K> -ctv <V> -p 512 -n 128 -r 1 -o md
```

| K cache | V cache | pp512 t/s | tg128 t/s | Notes |
|---|---:|---:|---:|---|
| f16 | f16 | 3507.12 | 120.59 | Baseline cache, no `type_k/type_v` columns emitted by bench because defaults |
| q8_0 | q8_0 | 3388.43 | 111.81 | Standard quantized KV |
| q4_0 | q4_0 | 3405.76 | 111.56 | Standard compressed KV |
| turbo4_0 | turbo3_0 | 1202.00 | 105.56 | TurboQuant path; lower pp due transform/compression overhead, strong long-context memory savings |

Interpretation: short-context prompt processing is fastest with normal KV cache types. TurboKV is valuable for fitting and sustaining large context windows, not for maximizing tiny-context pp512 throughput.

## Bonsai 8B Q2_0 full deterministic 64k-style run

Command:

```bash
./bin/llama-cli -m ~/AI/models/Ternary-Bonsai-8B-Q2_0.gguf -p "Write a detailed, structured technical essay about where snails live, covering habitats, climate, food sources, predators, reproduction, and ecological roles. Continue in clear paragraphs." -n 4096 -c 65535 -s 12345 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1.0 --ignore-eos --no-display-prompt --flash-attn on -ngl 40 -ctk turbo4_0 -ctv turbo3_0 --perf -st
```

Result:

```text
[ Prompt: 910.8 t/s | Generation: 99.6 t/s ]
CUDA0: 7840 = 2573 free + (4346 self = 1918 model + 2124 context + 304 compute) + 920 unaccounted MiB
Host: 301 = 157 model + 0 context + 144 compute MiB
```

Lyndon's independent run of the same workload produced `~96.5 t/s`, matching the earlier Prism TurboKV checkpoint.

## Qwen3.6 35B A3B Q5_K_M short 256k-context run

Command:

```bash
./bin/llama-cli -m ~/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf -p "Write a detailed, structured technical essay about where snails live, covering habitats, climate, food sources, predators, reproduction, and ecological roles. Continue in clear paragraphs." -ngl 99 --n-cpu-moe 35 --no-mmap --mlock -ctk turbo4_0 -ctv turbo3_0 --no-warmup -s 12345 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1.0 --ignore-eos --no-display-prompt --flash-attn on -n 1024 -c 256000 --perf -st
```

Result:

```text
[ Prompt: 109.2 t/s | Generation: 45.3 t/s ]
CUDA0: 7840 = 411 free + (6441 self = 4249 model + 1215 context + 977 compute) + 986 unaccounted MiB
Host: 19831 = 19323 model + 0 context + 508 compute MiB
```

Lyndon's longer 4096-token run on the same model/settings held around `43 t/s`, same or `1-2 t/s` better than the older `llama-cpp-turboquant` build.

## Takeaways

- The combined branch preserves Prism's Bonsai TurboKV performance while moving onto the newer upstream base.
- Q2_0 / ternary model support and TurboKV cache support work together.
- The large Qwen MoE path is not regressing versus the dedicated TurboQuant fork and may be slightly faster in this environment.
- Host KV context remains `0 MiB` in the tested runs; KV cache stays on GPU.
- `turbo4_0` K + `turbo3_0` V is the recommended proven path.
