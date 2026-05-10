# Turbo6 K Port — Handover File

**Created:** 2026-05-09 ~12:10 GMT+1  
**Updated:** 2026-05-09 ~21:35 GMT+1  
**Status:** Build succeeded, canary benchmarks completed  
**Target:** llama.cpp branch `main` in `~/AI/ternaryturboquant/llama.cpp`  
**Source:** `pitcany/ollama-turboquant` branch `origin/research/turbo5-turbo6-kernels` (commit `d7e40be2`)

---

## What We're Doing

Adding Turbo6 (6-bit PolarQuant) as a **KV cache dtype** (K-side only).  
Goal: pair with existing `turbo3_0` V → `turbo6_0 turbo3_0` mixed-KV preset that should beat `kq8-vturbo4` quality at lower KV memory.

---

## Files Modified (12 files, all in `ternaryturboquant/llama.cpp`)

| File | What Changed |
|------|--------------|
| `ggml/include/ggml.h` | Added `GGML_TYPE_TURBO6_0 = 46` enum + comment |
| `ggml/src/ggml-common.h` | Added `QK_TURBO6 = 128`, `QR_TURBO6 = 1`, `block_turbo6_0` struct, Turbo6 centroids, `turbo_nearest_centroid_6bit()` |
| `ggml/src/ggml-quants.h` | Added `dequantize_row_turbo6_0()`, `quantize_row_turbo6_0_ref()`, `quantize_turbo6_0()` prototypes |
| `ggml/src/ggml-quants.c` | Added `quantize_turbo6_0()` CPU quantizer + Turbo6 case in `compute_quant_error()` |
| `ggml/src/ggml-turbo-quant.c` | Added Turbo6 WHT functions (`turbo6_cpu_wht_group_size`, `turbo_cpu_fwht_inverse` for turbo6, `turbo_cpu_butterfly_fwht_inverse`) |
| `ggml/src/ggml.c` | Added `GGML_TYPE_TURBO6_0` type info block + quantize dispatch case |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Added `GGML_TYPE_TURBO6_0` traits (from_float → `quantize_row_turbo6_0_ref`), forward decl for `ggml_vec_dot_turbo6_0_f32`, Turbo6 case in `ggml_quantize_chunk()` |
| `ggml/src/ggml-cuda/dequantize.cuh` | Added `dequantize_turbo6_0()` device function |
| `ggml/src/ggml-cuda/turbo-quant.cuh` | Added Turbo6 centroids + nearest-centroid lookup |
| `ggml/src/ggml-cuda/convert.cu` | Added `GGML_TYPE_TURBO6_0` cases in `to_fp16_nc_cuda`, `to_fp32_nc_cuda`, `to_float_nc_cuda` |
| `ggml/src/ggml-cuda/set-rows.cu` | Added `k_set_rows_turbo6()` kernel + `set_rows_cuda_turbo6()` launcher + dispatch case |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Added Turbo6 KQ inner product (`vec_dot_fattn_vec_KQ_turbo6`) — vectorised 8 elements/iter, 6-bit indices via 48-bit fetch |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Added `DECL_FATTN_VEC_CASE` externs for turbo6_0+turbo3_0 (D=64,128,256) |
| `ggml/src/ggml-cuda/fattn.cu` | Added turbo6_0+turbo3_0 to `FATTN_VEC_CASES_ALL_D`, dispatch for Turbo6 in kernel selector, `is_kv_compat`, `any_turbo_kv`, `supported_turbo_vec` checks |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Added `fattn-vec-instance-turbo6_0-turbo3_0.cu` |
| `common/arg.cpp` | Added `GGML_TYPE_TURBO6_0` to supported turbo types list |

**Created 1 new file:**
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo6_0-turbo3_0.cu` — template instance for turbo6 K + turbo3 V

---

## Build

- **Status:** ✅ Clean build succeeded — `llama-server` and `llama-bench` built with no errors
- **Only warnings:** unused `warp_id` variables in turbo2/turbo3 set_rows (pre-existing), enum switch warnings (pre-existing)
- **Bug fix:** `llama-bench` has its own `ggml_type_from_name()` in `tools/llama-bench/llama-bench.cpp:471` with a hand-written lookup table. `turbo6_0` was missing (turbo2/3/4 were present). Added entry. Rebuild `llama-bench` after merge.

---

## Key Design Decisions

1. **K-only, V stays turbo3_0** — Turbo6 is a K-side quantisation. The mixed-KV preset is `turbo6_0 turbo3_0`.
2. **Block size = 128** — same as turbo3/turbo4, so it slots in cleanly with existing WHT group handling.
3. **6-bit centroids** — 64 quantised values (vs 16 for turbo4, 8 for turbo3) fetched via 48-bit integer (uint32 + uint16).
4. **Vectorised KQ dot product** — 8 elements per loop iteration (6 bytes per iter), byte-aligned fetches.
5. **set_rows kernel** — same 128-thread-per-block layout as turbo4; WHT butterfly + norm + 6-bit quant in one pass.

---

## Canary Benchmarks (Bonsai 8B Q2_0, RTX 3070 Ti 8GB, Flash Attn)

| Config | Prefill (t/s) | Token Gen (t/s) | Bits/K | Memory/K |
|---|---|---|---|---|
| q8_0 + turbo3_0 | 3372 ± 62 | 107.1 ± 0.2 | 8-bit | baseline |
| turbo4_0 + turbo3_0 | 3268 ± 63 | 104.1 ± 0.4 | 4-bit | 50% of q8_0 |
| **turbo6_0 + turbo3_0** | **3168 ± 134** | **98.8 ± 1.0** | **6-bit** | **75% of q8_0** |

**Key takeaway:** Turbo6 is ~3-5% slower than Turbo4, but uses 6 bits vs 4 bits per K element. The real win is vs q8_0: 25% less KV memory with ~8% throughput cost. This is the quality-vs-speed sweet spot for limited-VRAM long-context inference.

**Not tested on Qwen 35B** — the Q5_K_M model is 24GB and does not fit on 8GB VRAM. Designed for local-llama CPU server at localhost:8080.

## Pending

- [x] **Run canary tests** with `turbo6_0 turbo3_0` preset
  - ✅ Bonsai 8B Q2_0 benchmark complete (see above)
  - ❌ Qwen 35B — model too large for 8GB GPU
- [ ] **Quality comparison** — does Turbo6 K produce better quality vs turbo4 K in actual inference?
- [ ] **Bonsai `send_head` regression test** — run the same prompt through `llama-server` with turbo6 vs turbo4 and compare output quality
- [ ] **Memory cap test** — how many context tokens fit with `turbo6_0 turbo3_0` vs `turbo4_0 turbo3_0` before OOM on 8GB GPU?

---

## How to Test

```bash
# Turbo6 K + Turbo3 V (the target preset)
./build/bin/llama-server -m <model.gguf> -ct turbo6_0 turbo3_0

# Compare with current baseline
./build/bin/llama-server -m <model.gguf> -ct turbo4_0 turbo3_0
./build/bin/llama-server -m <model.gguf> -ct turbo4_0 turbo4_0

# llama-bench for perf numbers
./build/bin/llama-bench -m <model.gguf> -ct turbo6_0 turbo3_0
```

---

## Previous Session Context (for reference)

The compacted summary before this session covered:
- Turbo5/6 research path review: `pitcany/ollama-turboquant` branch `origin/research/turbo5-turbo6-kernels` was identified as the best port target
- Key insight: Turbo6 K in llama.cpp's format with turbo3 V should deliver better quality than turbo4 K at same or lower KV memory
- The ollama-turboquant fork already has working Turbo6 kernels (CPU + CUDA)
