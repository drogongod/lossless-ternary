# Lossless Base-3 / Base-5 Integer Arithmetic for Multiply-Free Inference

A small, verifiable number system: represent values in **base-3 (ternary)** or **base-5 (quinary)** with
**zero rounding** — a *change of base*, not a lossy quantization — plus multiply-free kernels, and the case for
running ternary/low-bit inference on the **CPU** instead of the GPU.

The whole thing is checked by exact equality (`allclose` with `atol=0`, and a framework-free C check). No
hand-waving: the tests either pass bit-exactly or they don't.

## The idea in one paragraph

Take a floating-point value, scale it by a fixed factor `S` into an exact 64-bit integer, then write that integer
in **positional base-B notation** (B = 3 or 5). Writing a number in a different base *cannot* lose information —
`10` in binary is `1010`, same number, different digits. Reconstruction is therefore exact. On top of that,
a **base-native matmul** computes the dot product from the digit-planes and their positional powers using only
integer add/subtract/shift, and it equals a plain integer GEMM **exactly**.

This is **not BitNet.** BitNet rounds each weight to `{-1, 0, +1}` and throws the rest of the number away (lossy).
Here nothing is thrown away — the base representation carries the whole number. The only precision floor is the
fixed-point scale `S`, which is a dial *you* set, not a rounding you suffer.

## Why ternary doesn't want a GPU

A GPU is a machine for doing enormous numbers of **floating-point multiplies** in parallel. Ternary weights
`{-1, 0, +1}` make the multiply *disappear*: the inner loop is `add / skip / subtract`. And ternary weights are
tiny (~1.58 bits each), so there is little to move across the memory bus. That leaves a workload that is both
**compute-light** and **memory-light** — exactly the workload a modern CPU with SIMD (AVX-512 / NEON) or a small
amount of hand-written assembly eats for breakfast. The GPU's float-multiply muscle is simply wasted here.

This is not a lone opinion: Microsoft's [`bitnet.cpp`](https://github.com/microsoft/BitNet) already runs ternary
LLMs on commodity CPUs at speeds competitive with, or better than, GPU inference. The direction is real.

## In the spirit of Egyptian arithmetic

Ancient Egyptian mathematics was **exact and additive**. Multiplication was done by **repeated doubling and
addition** (a shift-and-add scheme — no multiplication table), and fractions were written as **exact sums of unit
fractions** — no rounding, ever. This work is in that same spirit: represent numbers *exactly*, compute by
*addition* rather than multiplication, and refuse to round. Old ideas, meeting a modern need.

## Training the weights: Gretchen

The kernels above *run* ternary weights; **Gretchen** *makes* them. It's a LoRA that crystallizes its weights to
`{-1, 0, +1}` **during training** via a smooth polynomial well — **no rounding, no straight-through estimator**.
The ternary structure emerges from the loss landscape itself. See **[`GRETCHEN.md`](GRETCHEN.md)** for the well,
the zero-basin fix, and usage.

## What's here

| file | what it is |
|------|------------|
| `src/lossless_base.py` | lossless base-3/5 converter + verification (`torch.allclose`, `atol=0, rtol=0`) |
| `src/ternary_matmul.c` | framework-free C: base decompose/reconstruct + multiply-free ternary matmul, self-check `main()` |
| `src/gretchen.py` | **Gretchen** — train ternary weights via a polynomial well (no rounding, no STE) |
| `GRETCHEN.md` | how Gretchen works + the zero-basin fix |
| `EGYPTIAN.md` | the multiply-free / exact-additive rhyme with ancient Egyptian arithmetic |
| `src/fastice_kernel.c` | **fastICE** — real SIMD quinary (5-level) multiply-free kernel, beats OpenBLAS ~2.85-2.9x |
| `src/fastice_trainer.py` | full training step (forward+backward+Adam) wired through fastICE |
| `FASTICE.md` | fastICE in full — why a naive version loses, why the real SIMD version wins, a real bug found+fixed |
| `README.md` / `LICENSE` | this / PolyForm Noncommercial 1.0.0 |

Run the Python check:
```
python src/lossless_base.py
```
Compile and run the C check (no dependencies):
```
cc -O2 -o ternary src/ternary_matmul.c && ./ternary
```
Both print `PASS` only if every value round-trips bit-exactly and the multiply-free matmul equals the integer
reference.

## Honest scope

This is a **clean, verified implementation and a clear argument**, built on well-understood ideas — positional
notation, ternary/low-bit weights, and integer GEMM. The contribution is the *tidy, exact, framework-free package*
and the case for **CPU/assembly sovereignty**: you do not need CUDA, a specific vendor's GPU, or a heavyweight
framework to run low-bit models. Portable C compiles everywhere; the hot kernel drops to SIMD or NASM when you
want the last drop of speed. It runs on hardware you own, owes nothing to anyone, and the arithmetic is exact.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free for
any noncommercial purpose (personal, research, nonprofit, education, government). Commercial use is
not covered by this license — contact the author to arrange a commercial license.
