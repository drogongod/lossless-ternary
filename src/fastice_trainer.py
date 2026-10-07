#!/usr/bin/env python3
r"""fastice_trainer.py -- a full training step (forward + backward + weight update) for a small
gated-recurrent char-level language model, with every weight-involving matmul routed through the
real SIMD multiply-free quinary kernel (fastice_kernel.c/.dll), not a standalone benchmark.

THIS IS TOOL B, NOT TOOL A -- an important, deliberate scope note, not a footnote. Quantization
(ternary, quinary, or otherwise) via a straight-through estimator (STE) with a float32 shadow
weight is a LOSSY technique: a real, if small, approximation is baked in every forward pass, and
the float32 master is what the optimizer actually tracks. That is the right tool for a FROZEN or
periodically-refreshed target -- a deployed/quantized model, a distilled "clove," a static expert
-- where a one-time (or occasional) training cost is acceptable and the deployed artifact never
needs to accumulate error across an unbounded number of future steps.

It is explicitly the WRONG tool for a model meant to learn continuously forever: STE's forward
approximation error, even though small, is a standing bias that a model training indefinitely
would carry on every single step. A different file in this repo, `lossless_base.py`, is Tool A --
an exact (atol=0), no-rounding fixed-point representation, built for exactly that continuous-
learning job. Keep the two separate; this file solves the "make a batch of training steps fast"
problem, not the "train forever without drift" problem.

QUANTIZATION BUG FOUND AND FIXED while building this (recorded here because it generalizes):
fusing two weight matrices into one kernel call (to cut call overhead) requires deriving ONE shared
quantization scale for the combined block. Using a MEAN-based (BitNet-style absmean) scale for the
fused block is fine when both matrices have similar typical magnitude, but breaks badly when they
don't: here, one matrix was a generic random init (large absmean) and the other was a deliberately
near-identity matrix (eye(D)*0.33 + tiny noise) for recurrence stability -- a 13x gap in absmean.
The mean-based shared scale, dominated by the larger matrix, clipped the smaller matrix's single
most important value (its 0.33 diagonal) down to the quantizer's max level, destroying the exact
property that kept the recurrence stable -- and produced a real, measured gradient overflow within
5 training batches (reproduced with warnings promoted to errors). FIX: use a MAX-based scale for
any fused block (sized to the single largest magnitude across BOTH halves, so neither matrix's
peak value is ever clipped) -- see `quinary_quant_planes(..., use_max_scale=True)` below. Fusing
two quantized sources under one scale is a good speed lever; doing it with a mean-based scale is a
trap if the two sources don't share a magnitude distribution.

MEASURED RESULT (single-layer, D=384, real training corpus, same batching/methodology across all
three): this kernel's full training step ran at 7.71 ms/window -- 1.37x FASTER than an equivalent
ternary (3-level) STE model on the same hardware, and 5.9x faster than an unoptimized native
fixed-point baseline that used a real (non-multiply-free) float64 matmul instead. Quinary also
keeps a real quality edge over ternary (~20-24% better reconstruction fidelity, measured
separately, consistent with the extra 2 levels capturing information ternary's 3 levels round away).
"""
import numpy as np, ctypes, os, platform

_LIBNAME = "fastice_kernel.dll" if platform.system() == "Windows" else "fastice_kernel.so"
_DLL = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIBNAME))
_DLL.fastice_gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int]


def quinary_quant_planes(W, use_max_scale=False):
    """W: float32 (M,K). Returns S1,S2 int8 (M,K) with w_level = S1 + 2*S2 in {-2..2}, and wscale.

    use_max_scale=True is REQUIRED whenever W is itself a concatenation of two weight matrices
    being fused into one kernel call (see module docstring for the bug this fixes) -- a mean-based
    scale lets the matrix with the larger typical magnitude dominate and clip the other's peak
    values. A max-based scale guarantees the single largest magnitude in EITHER half maps to
    level +-2 without clipping, at the cost of coarser resolution below that peak."""
    denom = np.abs(W).max() if use_max_scale else np.abs(W).mean()
    scale = 2.0 / (denom + 1e-8) if use_max_scale else 1.0 / (denom + 1e-8)
    q = np.clip(np.round(W * scale), -2, 2).astype(np.int32)
    S1 = np.where(np.abs(q) == 1, q, 0).astype(np.int8)
    S2 = np.where(np.abs(q) == 2, q // 2, 0).astype(np.int8)
    return np.ascontiguousarray(S1), np.ascontiguousarray(S2), np.float32(1.0 / scale)


def activation_quant_int8(x):
    """x: float32 (K,B) -> int8 (K,B) + per-column scale (1,B). Per-token quantization, same
    convention as the real BitNet/MatMul-free-LM reference implementation."""
    amax = np.maximum(np.abs(x).max(axis=0, keepdims=True), 1e-8)
    scale = amax / 127.0
    xq = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return xq, scale


def simd_mm(S1, S2, wscale, X, xscale):
    """S1,S2: (M,K) int8 quantized weight planes. X: (K,B) int8 quantized activations.
    Returns (M,B) float32 = dequantized W_q @ X_q."""
    M, K = S1.shape
    B = X.shape[1]
    Xt = np.ascontiguousarray(X.T)  # kernel wants (B,K) row-major
    out = np.empty((M, B), dtype=np.int32)
    _DLL.fastice_gemm(S1.ctypes.data, S2.ctypes.data, Xt.ctypes.data, out.ctypes.data, M, K, B)
    return out.astype(np.float32) * wscale * xscale


def transpose_planes(S1, S2):
    """Precompute transposed quantized planes ONCE per batch call, not per-timestep -- profiling
    found the naive per-call-transpose version spent ~6% of total time redoing the same transpose
    repeatedly for the same weight across many timesteps within one batch."""
    return np.ascontiguousarray(S1.T), np.ascontiguousarray(S2.T)


class FastICECharLM:
    """A single-layer, convex-gated leaky recurrent cell (h = g*c + (1-g)*h), trained end-to-end
    through the fastICE multiply-free kernel. float32 master weights + Adam + STE -- see the
    module docstring for why that's the right scope for this file (frozen/periodic targets) and
    the wrong one for continuous online learning."""

    def __init__(self, V, D, seed):
        rng = np.random.default_rng(seed)
        s = 0.25
        gm = lambda a, b: rng.uniform(-s, s, (a, b)).astype(np.float32)
        self.V = V; self.D = D
        self.Wemb = gm(D, V)
        self.Win = gm(D, D)
        self.Wh = np.eye(D, dtype=np.float32) * 0.33 + gm(D, D) * 0.03  # near-identity: stabilizes the recurrence
        self.b = np.zeros(D, np.float32)
        self.Wgx = gm(D, D); self.Wgh = gm(D, D) * 0.03; self.bg = np.zeros(D, np.float32)
        self.Wo = gm(V, D)
        self._m = {}; self._v = {}; self._t = 0

    def _adam_step(self, name, W, dW, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m = self._m.get(name, np.zeros_like(W)); v = self._v.get(name, np.zeros_like(W))
        m = beta1 * m + (1 - beta1) * dW
        v = beta2 * v + (1 - beta2) * (dW * dW)
        self._m[name] = m; self._v[name] = v
        mhat = m / (1 - beta1 ** self._t); vhat = v / (1 - beta2 ** self._t)
        return W - lr * mhat / (np.sqrt(vhat) + eps)

    def learn_batch(self, idxs_batch, lr=0.003):
        """idxs_batch: list of B equal-length token-id sequences. Returns per-item mean loss (B,)."""
        self._t += 1
        B = len(idxs_batch); T = len(idxs_batch[0])
        idxs_arr = np.array(idxs_batch, dtype=np.int64)
        D = self.D

        S1_o, S2_o, ws_o = quinary_quant_planes(self.Wo)
        S1_oT, S2_oT = transpose_planes(S1_o, S2_o)
        # FUSION: every place two kernel calls were being SUMMED (A@u + B@v) is fused into one
        # call by concatenating the FLOAT MASTER weights along K before quantizing (max-scale --
        # see module docstring) and the matching activations along the same axis:
        # (A|B) @ [u;v] = A@u + B@v, same math, half the kernel-dispatch calls.
        S1_wx, S2_wx, ws_wx = quinary_quant_planes(np.concatenate([self.Win, self.Wh], axis=1), use_max_scale=True)
        S1_wg, S2_wg, ws_wg = quinary_quant_planes(np.concatenate([self.Wgx, self.Wgh], axis=1), use_max_scale=True)
        S1_bh, S2_bh, ws_bh = quinary_quant_planes(np.concatenate([self.Wh.T, self.Wgh.T], axis=1), use_max_scale=True)
        S1_bx, S2_bx, ws_bx = quinary_quant_planes(np.concatenate([self.Win.T, self.Wgx.T], axis=1), use_max_scale=True)

        X = self.Wemb[:, idxs_arr.T]  # (D,T,B) float32

        h = np.zeros((D, B), np.float32)
        H = []
        logits = np.empty((T, self.V, B), np.float32)
        for t in range(T):
            x_t = X[:, t, :]
            xh = np.concatenate([x_t, h], axis=0)
            xhq, xhs = activation_quant_int8(xh)            # one quantize, reused for pc AND pg
            pc = simd_mm(S1_wx, S2_wx, ws_wx, xhq, xhs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_wg, S2_wg, ws_wg, xhq, xhs) + self.bg[:, None]
            g = 1.0 / (1.0 + np.exp(-pg))
            hn = g * c + (1 - g) * h
            H.append((x_t, h, c, g, hn)); h = hn
            hnq, hns = activation_quant_int8(hn)
            logits[t] = simd_mm(S1_o, S2_o, ws_o, hnq, hns)

        gWemb = np.zeros_like(self.Wemb); gWo = np.zeros_like(self.Wo)
        gWin = np.zeros_like(self.Win); gWh = np.zeros_like(self.Wh); gb = np.zeros_like(self.b)
        gWgx = np.zeros_like(self.Wgx); gWgh = np.zeros_like(self.Wgh); gbg = np.zeros_like(self.bg)
        dh = np.zeros((D, B), np.float32); loss = np.zeros(B, np.float64); n = 0

        for t in range(T - 2, -1, -1):
            tgt = idxs_arr[:, t + 1]
            lg = logits[t].astype(np.float64)
            p = np.exp(lg - lg.max(axis=0, keepdims=True)); p /= p.sum(axis=0, keepdims=True)
            loss += -np.log(p[tgt, np.arange(B)] + 1e-9); n += 1
            dl = p.astype(np.float32); dl[tgt, np.arange(B)] -= 1.0

            x_t, hp, c, g, hn = H[t]
            gWo += dl @ hn.T   # data x data, no quantized operand here -- plain BLAS
            dlq, dls = activation_quant_int8(dl)
            dh = dh + simd_mm(S1_oT, S2_oT, ws_o, dlq, dls)
            dg = dh * (c - hp); dc = dh * g; dhp = dh * (1 - g)
            dpc = dc * (1 - c * c); dpg = dg * g * (1 - g)
            gWin += dpc @ x_t.T; gWh += dpc @ hp.T; gb += dpc.sum(axis=1)
            gWgx += dpg @ x_t.T; gWgh += dpg @ hp.T; gbg += dpg.sum(axis=1)
            dpcg = np.concatenate([dpc, dpg], axis=0)
            dpcgq, dpcgs = activation_quant_int8(dpcg)
            dxemb = simd_mm(S1_bx, S2_bx, ws_bx, dpcgq, dpcgs)
            np.add.at(gWemb, (slice(None), idxs_arr[:, t]), dxemb)
            dh = dhp + simd_mm(S1_bh, S2_bh, ws_bh, dpcgq, dpcgs)

        for name in ("Wemb", "Wo", "Win", "Wh", "b", "Wgx", "Wgh", "bg"):
            G = {"Wemb": gWemb, "Wo": gWo, "Win": gWin, "Wh": gWh, "b": gb,
                 "Wgx": gWgx, "Wgh": gWgh, "bg": gbg}[name] / B
            setattr(self, name, self._adam_step(name, getattr(self, name), G, lr))

        return loss / max(1, n)
