# Gretchen — training ternary weights with a polynomial well (no rounding, no STE)

**Gretchen** is a LoRA variant whose adapter weights **crystallize to ternary `{-1, 0, +1}` during training** —
not by rounding them at the end, and not with a straight-through estimator, but by adding a smooth **polynomial
well** to the loss that pulls each weight into one of three basins. The ternary structure emerges from the loss
landscape itself, so there is never a rounding step and never a gradient you have to fake.

## The well

```
P(w) = w^6 - 2w^4 + w^2        # minima at -1, 0, +1 (all zero-energy)
```

Added to the task loss as a regularizer (scaled by λ), it charges an energy cost to any weight that *isn't* near
−1/0/+1, so gradient descent settles the weights into the wells. It is fully differentiable — **no rounding, no
STE, no step functions.**

## The zero-basin fix

The naïve well is *indifferent* between 0 and ±1 (all three are zero-energy), so the optimizer takes the cheapest
route and collapses **everything to 0** — a dead adapter that does nothing. This is a real failure mode, and the
fix is three-fold:

1. **Lower the LoRA scale (alpha)** so the task is only satisfied by *larger* raw weights, pushing them outward.
2. **Bimodal init** — start weights near ±0.5, already inside the ±1 basins of attraction, instead of pooled at 0.
3. **Anti-zero bump** — a narrow Gaussian that *raises* energy at 0, so ±1 become the only free minima.

Watch the basin monitor fill over training: `-1% / 0% / +1%`. If 0% stays ~100 by mid-training, the pressure is
too weak — raise λ or the anti-zero penalty. (Verify the histogram; don't assume it worked.)

## Why it matters

Ternary weights are **multiply-free**: −1/0/+1 means every operation is *subtract / skip / add*, with no
floating-point multiply anywhere. So a Gretchen-trained adapter runs with **no GPU and no CUDA** — on a CPU, in
SIMD or hand-written assembly. It pairs directly with the lossless base-3/5 kernels in this repo: Gretchen makes
the ternary weights; those kernels run them.

## Honest scope

Regularizing weights toward discrete values is a known idea. Gretchen's contribution is a **clean, specific,
working** instance of it: the particular triple-well, the fully-continuous (no-STE) crystallization, and a
documented fix for the zero-basin collapse these methods hit in practice. It's offered as a usable tool and a
clear write-up — not a claim to have invented quantization-aware training.

## Source & usage

`src/gretchen.py` is self-contained (PyTorch): the well, the Gretchen LoRA module, injection into a model, a
basin monitor, and chunked save/load. A training loop uses it like this:

```python
from gretchen import inject_gretchen, gretchen_params, gretchen_well_loss, basin_report

inject_gretchen(model, rank=16, alpha=4.0, init_mode="bimodal", init_scale=0.5)
opt = torch.optim.AdamW(gretchen_params(model), lr=2e-4)

for step, (x, y) in enumerate(loader):
    task = loss_fn(model(x), y)
    well = gretchen_well_loss(model, lam=lambda_schedule(step), zero_penalty=0.05)
    (task + well).backward(); opt.step(); opt.zero_grad()
    if step % 50 == 0:
        print(basin_report(model))   # -> {'std':.., 'zero%':.., 'neg1%':.., 'pos1%':..}
```

Author: **Jonathan David Wint**.  License: [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (free for noncommercial use; commercial use requires a separate license — contact the author).
