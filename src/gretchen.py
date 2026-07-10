#!/usr/bin/env python3
"""
gretchen_v7.py  — V7 (zero-basin fix)
Author: Jonathan David Wint (Azazel) & Claude
Date: June 2026

V6 POST-MORTEM (why this rewrite exists):
  V6 trained cleanly, loss fell ~11 -> 1.6, gradient flowed (zero-trap fix held).
  BUT the basin diagnostic showed 100% of raw A/B weights sitting in the 0 well,
  0% reaching +/-1. The crystallization SUCCEEDED toward the WRONG target.
  Root cause = three compounding pulls toward zero:
    1) scale = alpha/r = 32/16 = 2.0 is LARGE, so the task is satisfied with tiny
       raw weights (~0.01); the well then catches those tiny weights at 0.
    2) init randn*0.01 starts every weight microscopically close to the 0 basin.
    3) the potential P(w)=w^6-2w^4+w^2 is INDIFFERENT between 0 and +/-1 (both have
       zero energy), so the optimizer takes the cheapest route: 0.

V7 FIXES (all tunable; defaults chosen to push toward +/-1 without re-causing collapse):
    A) alpha default 32 -> 4   (scale 2.0 -> 0.25): network must grow raw weights
       ~8x larger to hit the same dW, pushing them out toward +/-1.
    B) init: wider, optionally bimodal near +/-1 so weights START in the +/-1 basins
       of attraction instead of sitting in the 0 well from step 0.
    C) NEW: an optional anti-zero term in the well so 0 is no longer free -- the
       potential is tilted to PREFER +/-1 over 0. Applied by the TRAINER via
       gretchen_well_loss(), not inside forward().

IMPORTANT: the polynomial well is applied as a LOSS term in the training loop.
This file provides the well functions so the trainer can call them. If your trainer
computes the well inline, replace that with a call to gretchen_well_loss().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

# -----------------------------------------------------------------------------
# THE WELL  (now with an optional anti-zero tilt)
# -----------------------------------------------------------------------------
def well_potential(w: torch.Tensor, zero_penalty: float = 0.0) -> torch.Tensor:
    """
    Base manifesto potential: P(w) = w^6 - 2w^4 + w^2  (minima at -1, 0, +1, all =0).

    zero_penalty > 0 ADDS a bump at 0 so the 0 basin is no longer free:
        +zero_penalty * exp(-(w^2)/(2*0.1^2))
    A narrow Gaussian centered at 0 that RAISES energy near zero, leaving the +/-1
    wells untouched. Result: +/-1 become the only zero-energy minima, pushing the
    optimizer OUT of the lazy zero basin toward +/-1.
    """
    base = w**6 - 2*w**4 + w**2
    if zero_penalty > 0.0:
        bump = zero_penalty * torch.exp(-(w**2) / (2 * 0.1**2))
        return base + bump
    return base

def gretchen_well_loss(model: nn.Module, lam: float, zero_penalty: float = 0.0) -> torch.Tensor:
    """Mean well energy across all Gretchen A/B params, scaled by lam. Add to task loss."""
    total = None
    count = 0
    for n, p in model.named_parameters():
        if (n.endswith(".A") or n.endswith(".B")) and p.requires_grad:
            e = well_potential(p, zero_penalty).sum()
            total = e if total is None else total + e
            count += p.numel()
    if count == 0:
        return torch.zeros((), device=next(model.parameters()).device)
    return lam * (total / count)

# -----------------------------------------------------------------------------
# THE MODULE
# -----------------------------------------------------------------------------
class Gretchen(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 4.0,
                 init_mode: str = "wide", init_scale: float = 0.5):
        """
        alpha default 4.0 (was 32.0) -> scale 0.25 (was 2.0): forces larger raw weights.
        init_mode:
          "wide"    -> randn * init_scale (default 0.5): starts spread, not pinned at 0.
          "bimodal" -> weights start near +/-init_scale (in the +/-1 basins of attraction).
          "tiny"    -> the OLD randn*0.01 behavior (kept for comparison / resume).
        """
        super().__init__()
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.rank         = rank
        self.scaling      = alpha / rank
        self.has_bias     = base.bias is not None

        self.register_buffer("W", base.weight.data.float().clone())
        if self.has_bias:
            self.register_buffer("b", base.bias.data.float().clone())
        else:
            self.b = None

        self.A = nn.Parameter(self._init(rank, self.in_features, init_mode, init_scale))
        self.B = nn.Parameter(self._init(self.out_features, rank, init_mode, init_scale))

    @staticmethod
    def _init(d0, d1, mode, s):
        if mode == "tiny":
            return torch.randn(d0, d1, dtype=torch.float32) * 0.01
        if mode == "bimodal":
            signs = (torch.randint(0, 2, (d0, d1), dtype=torch.float32) * 2 - 1)
            jitter = torch.randn(d0, d1, dtype=torch.float32) * 0.05
            return signs * s + jitter
        return torch.randn(d0, d1, dtype=torch.float32) * s  # "wide"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x32 = x.float()
        W_fp32 = self.W.to(torch.float32)
        b_fp32 = self.b.to(torch.float32) if self.b is not None else None
        frozen = F.linear(x32, W_fp32, b_fp32)
        lora_delta = (x32 @ self.A.to(torch.float32).t() @ self.B.to(torch.float32).t()) * self.scaling
        out = frozen + lora_delta
        return out.to(original_dtype)

    def merge_weights(self) -> nn.Linear:
        merged_W = self.W + (self.B @ self.A) * self.scaling
        linear = nn.Linear(self.in_features, self.out_features, bias=self.has_bias)
        linear.weight.data = merged_W.to(self.W.dtype)
        if self.has_bias:
            linear.bias.data = self.b.to(self.W.dtype)
        return linear

# -----------------------------------------------------------------------------
# INJECTION & PARAMS
# -----------------------------------------------------------------------------
TARGETS = {"q", "k", "v", "o", "gate", "up", "down"}

def inject_gretchen(model: nn.Module, rank: int = 16, alpha: float = 4.0,
                    init_mode: str = "wide", init_scale: float = 0.5) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf in TARGETS and isinstance(module, nn.Linear):
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1],
                    Gretchen(module, rank=rank, alpha=alpha,
                             init_mode=init_mode, init_scale=init_scale))
            replaced += 1
    return replaced

def gretchen_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if (n.endswith(".A") or n.endswith(".B")) and p.requires_grad]

# -----------------------------------------------------------------------------
# BASIN MONITOR -- call every N steps to WATCH the +/-1 basins fill
# -----------------------------------------------------------------------------
def basin_report(model: nn.Module) -> dict:
    ws = []
    for n, p in model.named_parameters():
        if (n.endswith(".A") or n.endswith(".B")) and p.requires_grad:
            ws.append(p.detach().flatten())
    if not ws:
        return {}
    w = torch.cat(ws).float()
    near_0  = ((w >= -0.5) & (w <= 0.5)).float().mean().item()*100
    near_m1 = ((w > -1.5) & (w < -0.5)).float().mean().item()*100
    near_p1 = ((w > 0.5) & (w < 1.5)).float().mean().item()*100
    return {"std": round(w.std().item(),4), "zero%": round(near_0,2),
            "neg1%": round(near_m1,2), "pos1%": round(near_p1,2)}

# -----------------------------------------------------------------------------
# SAVE / LOAD  (unchanged keys: "{name}.A" / "{name}.B")
# -----------------------------------------------------------------------------
def save_gretchen(model: nn.Module, path: str) -> int:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, Gretchen):
            state[f"{name}.A"] = module.A.data.cpu().float()
            state[f"{name}.B"] = module.B.data.cpu().float()
    torch.save(state, path)
    return len(state) // 2

def load_gretchen(model: nn.Module, path: str) -> int:
    if not Path(path).exists():
        print(f" Gretchen: {path} not found. Skipping load.")
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f" Gretchen: Failed to load {path}. Error: {e}")
        return 0
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, Gretchen):
            kA, kB = f"{name}.A", f"{name}.B"
            if kA in state and kB in state:
                module.A.data.copy_(state[kA].float())
                module.B.data.copy_(state[kB].float())
                loaded += 1
            else:
                print(f" Gretchen: Missing keys for {name}")
    print(f" Gretchen: Loaded {loaded} layers from {path}")
    return loaded

def check_gretchen(model: nn.Module) -> dict:
    problems = {}
    for name, module in model.named_modules():
        if isinstance(module, Gretchen):
            issues = []
            if torch.isnan(module.W).any(): issues.append("W has NaN")
            if torch.isnan(module.A).any(): issues.append("A has NaN")
            if torch.isnan(module.B).any(): issues.append("B has NaN")
            if issues: problems[name] = issues
    return problems

def count_gretchen(model: nn.Module) -> dict:
    layers = sum(1 for _, m in model.named_modules() if isinstance(m, Gretchen))
    params = sum(p.numel() for p in gretchen_params(model))
    return {"layers": layers, "trainable_params": params}

# -----------------------------------------------------------------------------
# CHUNKED SAVE / LOAD  (required by sovereign_train_v7.py — one file, many chunks)
# keys: "{chunk_name}.{name}.A" / ".B"
# -----------------------------------------------------------------------------
def save_gretchen_chunked(chunk_name: str, model: nn.Module, path: str) -> int:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, Gretchen):
            state[f"{chunk_name}.{name}.A"] = module.A.data.cpu().float()
            state[f"{chunk_name}.{name}.B"] = module.B.data.cpu().float()
    if Path(path).exists():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=True)
            existing.update(state)
            state = existing
        except Exception as e:
            print(f" Gretchen: failed to merge existing chunk file, overwriting ({e})")
    torch.save(state, path)
    return len(state) // 2

def load_gretchen_chunked(chunk_name: str, model: nn.Module, path: str) -> int:
    if not Path(path).exists():
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f" Gretchen: failed to load chunk state {path}: {e}")
        return 0
    prefix = f"{chunk_name}."
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, Gretchen):
            kA, kB = f"{prefix}{name}.A", f"{prefix}{name}.B"
            if kA in state and kB in state:
                module.A.data.copy_(state[kA].float())
                module.B.data.copy_(state[kB].float())
                loaded += 1
    return loaded
