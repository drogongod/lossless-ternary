# In the spirit of Egyptian arithmetic

Ancient Egyptian mathematics had two habits that are exactly what low-bit / ternary inference wants. This is a
conceptual **rhyme**, not a claim of lineage — the Egyptians did not invent LLMs, and this is not literally their
math. But the *principles* are the same, and they're worth naming.

## 1. Multiply by doubling and adding — no multiplication table

To compute 13 × 17, an Egyptian scribe doubled 17 repeatedly (17, 34, 68, 136, …) and summed only the doublings
matching 13 = 8 + 4 + 1 → 136 + 68 + 17 = **221**. Multiplication done entirely with **shifts and additions**.

That is precisely what a ternary weight buys you: `{-1, 0, +1}` turns every multiply into *subtract / skip / add*.
A transformer's matmul is billions of multiplies; make the weights ternary and each one becomes an Egyptian
doubling-and-adding. The base-3/5 decomposition in this repo is the same move generalized — the Egyptians shifted
in base-2 (doubling); we shift in base-3 or base-5.

## 2. Represent exactly, by addition, and never round

Egyptian fractions were sums of distinct unit fractions (3/4 = 1/2 + 1/4); their numerals were additive (symbols
summed). Every quantity was an **exact additive decomposition** — no decimals, no rounding, ever. That is the soul
of the lossless base-3/5 representation here: a number written as an exact sum of digit × place-value. Represent
exactly, by addition; refuse to round.

## The point

An "Egyptian-math" model is one whose weights are ternary, so its arithmetic is done the way the pyramid-builders
did it — **multiply-free, additive, exact.** Modern hardware took the other road: enormous silicon devoted to
floating-point *multiply* (that is what a GPU is), and we followed. Ternary walks back to the older, exact,
additive way — and that way runs beautifully on a plain CPU. The Egyptians were compute-poor and got it right;
we got compute-rich and got it lazy.
