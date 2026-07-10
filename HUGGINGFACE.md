---
license: mit
tags:
  - ternary
  - quantization
  - cpu-inference
  - multiply-free
  - lora
language:
  - en
---

# Lossless Base-3/5 + Gretchen — a multiply-free ternary toolkit

Two pieces that fit together:

- **Gretchen** — train LoRA weights that crystallize to ternary `{-1, 0, +1}` *during* training, via a smooth
  polynomial well. No rounding, no straight-through estimator. (`gretchen.py`, `GRETCHEN.md`)
- **Lossless base-3/5 kernels** — represent values exactly in base-3/5 (a *change of base*, zero rounding) and
  run ternary matmuls multiply-free: `add / skip / subtract`, no floating-point multiply, no GPU, no CUDA. Proven
  bit-exact in Python (`torch.allclose`, `atol=0`) and in framework-free C. (`lossless_base.py`, `ternary_matmul.c`)

The case: a ternary model's matmul is compute-light *and* memory-light, so it belongs on a **CPU with SIMD or
hand-written assembly**, not a GPU built for floating-point multiplies. (Cf. Microsoft's `bitnet.cpp`.) In the
spirit of ancient Egyptian arithmetic — multiply by shift-and-add, represent exactly by addition, never round.

**Code, tests, and full write-ups:** https://github.com/<your-username>/lossless-ternary

Author: **Jonathan David Wint**. License: MIT.

> Note: this repo is a *method + toolkit* (code), not trained weights. A demo ternary model card can be added here
> once a Gretchen-trained checkpoint is released.
