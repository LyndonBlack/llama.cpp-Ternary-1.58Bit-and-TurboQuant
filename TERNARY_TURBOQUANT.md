# Ternary 1.58-bit + TurboQuant fork

This fork/branch combines PrismML Q2_0 / ternary model support with TurboQuant KV cache support and CUDA FlashAttention optimizations.

GitHub fork:

<https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>

Start here:

- Integration and usage notes: [`docs/turboternary/README.md`](docs/turboternary/README.md)
- Benchmark report: [`docs/turboternary/BENCHMARKS_2026-05-07.md`](docs/turboternary/BENCHMARKS_2026-05-07.md)
- Commit/integration summary: [`docs/turboternary/INTEGRATION_SUMMARY.md`](docs/turboternary/INTEGRATION_SUMMARY.md)
- Raw benchmark outputs: [`benchmark-results/2026-05-07-turboternary/`](benchmark-results/2026-05-07-turboternary/)

Known-good runtime path:

```bash
-ctk turbo4_0 -ctv turbo3_0 --flash-attn on
```
