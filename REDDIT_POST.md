**Title:** Lossless base-3/base-5 number system for multiply-free inference — and a case that ternary models don't need a GPU (working code + exact tests)

---

I put together a small, fully-verifiable thing and I've got an argument attached to it. Curious what people
here think, and what I'm missing.

**The thing.** A number system that writes values in base-3 (ternary) or base-5 (quinary) with **zero rounding.**
Not BitNet — BitNet *rounds* each weight to {-1,0,+1} and discards the rest (lossy). This is a **change of base**:
scale a float into an exact integer, write that integer in base-3/5 positional digits, reconstruct it perfectly.
Writing a number in a different base can't lose information — `10` is `1010` in binary, same number. On top of
that there's a **multiply-free matmul** for ternary weights that equals a plain integer GEMM *exactly*.

**The argument — ternary belongs on the CPU, not the GPU.** A GPU is a machine for doing huge numbers of
floating-point *multiplies* in parallel. Ternary weights {-1,0,+1} make the multiply *vanish* — the inner loop is
`add / skip / subtract`. And the weights are ~1.58 bits, so there's almost nothing to move across the memory bus.
Compute-light **and** memory-light = exactly what a modern CPU with SIMD (AVX-512/NEON) or a bit of hand-written
assembly is great at. The GPU's float-multiply muscle is just wasted. This isn't a lone-crank take, either —
Microsoft's `bitnet.cpp` already runs ternary LLMs on commodity CPUs at speeds competitive with GPUs.

**In the spirit of Egyptian arithmetic.** The Egyptians multiplied by *doubling and adding* (shift-add, no
multiplication table) and wrote fractions as *exact* unit-fraction sums (no rounding). Same spirit here: represent
exactly, compute by addition, refuse to round. Old ideas meeting a modern need.

**Proof, because claims are cheap.** The Python checks with `torch.allclose(atol=0, rtol=0)` — bit-exact. The C
version has no dependencies at all and prints `PASS` only if every value round-trips exactly and the multiply-free
matmul equals the integer reference. If they don't match, they say `FAIL`. Nothing to take on faith.

**Honest about scope.** This builds on well-understood ideas — positional notation, ternary/low-bit weights,
integer GEMM. I'm not claiming to have invented base-3. What I think is worth sharing is the *tidy, exact,
framework-free package* and the concrete case for **CPU/assembly sovereignty**: you don't need CUDA, a specific
vendor's GPU, or a heavy framework to run low-bit models. Portable C runs everywhere; the hot kernel drops to
SIMD/NASM when you want the last of the speed.

Repo (MIT): `[link]`

Genuine question for the people already doing low-bit CPU inference: is anyone working on *lossless* (not
lossy-quant) low-bit representations + hand-written kernels, and what are the pitfalls I should expect? What am
I missing?
