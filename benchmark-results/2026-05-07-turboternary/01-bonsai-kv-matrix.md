## llama-bench KV matrix, Bonsai d=0
\n### -ctk f16 -ctv f16
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7840 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes, VRAM: 7840 MiB
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |  1 |           pp512 |       3507.12 ± 0.00 |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |  1 |           tg128 |        120.59 ± 0.00 |

build: cb4bc74e2 (9059)
\n### -ctk q8_0 -ctv q8_0
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7840 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes, VRAM: 7840 MiB
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           pp512 |       3388.43 ± 0.00 |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |   q8_0 |   q8_0 |  1 |           tg128 |        111.81 ± 0.00 |

build: cb4bc74e2 (9059)
\n### -ctk q4_0 -ctv q4_0
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7840 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes, VRAM: 7840 MiB
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           pp512 |       3405.76 ± 0.00 |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 |   q4_0 |   q4_0 |  1 |           tg128 |        111.56 ± 0.00 |

build: cb4bc74e2 (9059)
\n### -ctk turbo4_0 -ctv turbo3_0
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7840 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes, VRAM: 7840 MiB
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 | turbo4_0 | turbo3_0 |  1 |           pp512 |       1202.00 ± 0.00 |
| qwen3 8B Q2_0                  |   2.03 GiB |     8.19 B | CUDA       |  99 | turbo4_0 | turbo3_0 |  1 |           tg128 |        105.56 ± 0.00 |

build: cb4bc74e2 (9059)
