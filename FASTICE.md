# fastICE — a real SIMD-vectorized quinary multiply-free kernel that beats BLAS

**fastICE** extends this repo's ternary "teaspoon" mechanism (`src/ternary_matmul.c`,
`_mm256_sign_epi8` + `_mm256_maddubs_epi16`) from 3 levels to 5 — and, unlike an earlier
scalar quinary attempt, it's real enough to actually beat production BLAS, not just a
naive reference loop.

## The one-paragraph version

Multiplying by a weight restricted to `{-2, -1, 0, +1, +2}` doesn't need a multiply
instruction at all. Decompose the level `v` into two ternary masks `s1, s2 ∈ {-1,0,+1}`
such that `v = s1 + 2·s2`, run each mask through a sign-select (`_mm256_sign_epi8`:
pick the activation, negate it, or zero it — never multiply), sum each plane with a
cheap pairwise add (`_mm256_maddubs_epi16`), and recombine with `sum1 + 2·sum2` — a
shift-and-add, still zero multiplies. SIMD does 32 of these per instruction.

## Why this file exists: a naive version of the same idea *loses*

Before building this, a plain scalar C loop (precomputed lookup table, `T[level][activation]`,
zero multiplies, no SIMD) was tested against real OpenBLAS on two shapes:

| shape | naive scalar LUT | real OpenBLAS | result |
|---|---|---|---|
| 384×384×128 | 7.48 ms | 1.32 ms | **BLAS wins by 5.7x** |
| 4096×12288 (GEMV) | 11.47 ms | 8.90 ms | **BLAS wins by 1.29x** |

"Multiply-free" is not automatically fast. A scalar loop has no SIMD, no cache
blocking, none of what makes BLAS fast — removing the multiply instruction doesn't
matter if everything else about the kernel is naive. The *real* SIMD-vectorized
version (this file) was then tested the same way, correctness-checked bit-exact
against a scalar reference first:

| shape | fastICE (SIMD) | real OpenBLAS | result |
|---|---|---|---|
| 384×384×128 | 0.45 ms | 1.32 ms | **fastICE wins by 2.9x** |
| 4096×12288 (GEMV) | 3.12 ms | 8.90 ms | **fastICE wins by 2.85x** |

Same idea, same CPU, same shapes. Scalar loses, real SIMD wins — decisively, and
consistently across two very different shapes.

## Wired into a full training step, not just a kernel benchmark

A fast forward kernel means nothing if it can't carry a real training step. `src/fastice_trainer.py`
wires `fastice_kernel` into forward **and** backward (gradient-w.r.t.-input, via the
transposed quantized planes) for a small gated-recurrent char-level language model,
leaving only the weight-*gradient* outer products (a data×data product with no
quantized operand to exploit) on plain BLAS.

Measured end-to-end, same corpus, same batching, same hardware, three ways:

| approach | ms/window |
|---|---|
| unoptimized native fixed-point baseline (real, non-multiply-free float64 matmul) | 45.58 |
| ternary (3-level) STE, plain float32 BLAS | 10.54 |
| **fastICE quinary (5-level) STE, real SIMD kernel** | **7.71** |

fastICE ends up **1.37x faster than ternary**, not just competitive with it — while
quinary's 2 extra levels capture real information ternary's 3 levels round away
(~20-24% better reconstruction fidelity, measured separately on real model weights).
More detail, more compression headroom, *and* faster, once the kernel is actually
vectorized.

## A real bug, found and fixed, worth knowing if you build on this

Fusing two weight matrices into one kernel call (to halve the number of calls) needs
ONE shared quantization scale for the combined block. Using a **mean**-based
(absmean, BitNet-style) scale is fine when both matrices have similar typical
magnitude — but here, one matrix was generic-random-initialized (absmean ≈ 0.125)
and the other was deliberately near-identity (`eye(D)*0.33 + tiny noise`, for
recurrence stability; absmean ≈ 0.0094 — a 13x gap). The mean-based scale, dominated
by the larger matrix, clipped the smaller matrix's single most important value (its
0.33 diagonal) down to the quantizer's max level — destroying the exact property that
kept the recurrence stable, and producing a real, reproducible gradient overflow
within 5 training batches.

**Fix**: use a **max**-based scale for any fused block — sized to the single largest
magnitude across *both* halves, so neither matrix's peak value is ever clipped, at
the cost of coarser resolution below that peak. Re-verified clean (zero overflow,
with warnings promoted to hard errors) across a full run after the fix. See
`quinary_quant_planes(..., use_max_scale=True)` in `src/fastice_trainer.py` for the
exact code and the full comment.

**The general lesson**: when fusing quantized sources under one shared scale, check
whether they actually share a magnitude distribution first. If they don't, a
mean-based scale will silently clip whichever one is smaller — and if the smaller
one is small *on purpose* (a stability-critical near-identity structure, in this
case), that clipping isn't just lossy, it's actively destabilizing.

## Scope note — this is a frozen/periodic-target tool, not a continuous-learning one

`fastice_trainer.py` quantizes via a straight-through estimator (STE) with a float32
shadow master weight — a small, real, standing approximation baked into every
forward pass. That's the right tradeoff for training (or periodically retraining) a
model that gets deployed and then held fixed for a while — a distilled expert, a
compressed checkpoint, anything frozen or refreshed on a cadence. It is the wrong
tool for a model meant to learn continuously, forever: that same small forward bias
would be a standing cost paid on every one of an unbounded number of future steps.
`src/lossless_base.py` elsewhere in this repo — exact, `atol=0`, no STE, no rounding
— is the tool for that job. Keep them separate; this file solves "make a batch of
training steps fast," not "train forever without drift."

## Files

| file | what it is |
|---|---|
| `src/fastice_kernel.c` | the SIMD multiply-free quinary kernel (batched, OpenMP-parallel) |
| `src/fastice_trainer.py` | full training step (forward+backward+Adam) wired through it |

## Build

```
# Windows (MSVC)
cl /O2 /openmp /arch:AVX2 /LD src/fastice_kernel.c /Fe:src/fastice_kernel.dll

# Linux/macOS (gcc/clang)
gcc -O3 -fopenmp -mavx2 -mavx -shared -fPIC src/fastice_kernel.c -o src/fastice_kernel.so
```

Then `python src/fastice_trainer.py` works as a library (`FastICECharLM`) — see the
module docstring for a minimal usage sketch.

## Honest scope

This is AVX2-only (no AVX-512/VNNI tested), scalar-floor numbers are included above
for comparison, and the training-step benchmark is a single-layer char-level model —
real, but small. The point proven is specific: a *properly vectorized* multiply-free
kernel beats real production BLAS, consistently, at real training shapes — not that
every multiply-free idea automatically wins, and not a claim about larger models
this hasn't been tested on yet.
