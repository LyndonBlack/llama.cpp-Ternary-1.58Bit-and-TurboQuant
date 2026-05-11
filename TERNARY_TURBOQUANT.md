# Ternary 1.58-bit + TurboQuant + Entropy-Adaptive KV Cache fork

This fork/branch combines PrismML Q2_0 / ternary model support with TurboQuant KV cache types, CUDA FlashAttention optimizations, and SCJedi entropy-adaptive per-layer mixed-precision K types.

GitHub fork:
<https://github.com/LyndonBlack/llama.cpp-Ternary-1.58Bit-and-TurboQuant>

Full documentation: [`docs/turboternary/README.md`](docs/turboternary/README.md)
- Entropy-adaptive KV cache (Path B) guide
- CodeNeedle validation results (all models)
- Recommended command lines for Bonsai 8B, Qwen3.6 35B A3B, Qwen3-VL-30B A3B
- Vision model support with --mmproj
- Calibration pipeline instructions
- Future research and "Mostlysane Local AI" web app

Known-good runtime path:
```bash
-ctk q8_0 -ctv turbo3_0 --flash-attn on
```
