#!/usr/bin/env python3
r"""
lossless_base_decompose.py -- Jonathan David Wint (Azazel)'s LOSSLESS ternary(base-3)/quinary(base-5) substrate.
NOT BitNet: no sign(), no rounding to {-1,0,+1}, no information thrown away. Instead:
  float -> EXACT int64 fixed-point (scale S) -> POSITIONAL base-B digit decomposition (a change of base = ZERO
  loss) -> perfect reconstruction -> a base-NATIVE matmul that computes the dot product from the DIGIT PLANES
  and their positional powers (integer accumulation, no float GEMM). Math is math.
The base decomposition + the base matmul add EXACTLY zero error (proved by allclose ==, not approx). The only
precision floor is the fixed-point scale S -- a dial you set, not a rounding you suffer.
"""
import torch

def to_fixed(x, S):
    "float -> exact int64 fixed-point (round to nearest; sign carried in the int)."
    return torch.round(x.to(torch.float64) * S).to(torch.int64)

def from_fixed(n, S):
    return n.to(torch.float64) / S

def digit_width(max_abs, B):
    w, cap = 1, B
    while cap <= max_abs: cap *= B; w += 1
    return w

def decompose(n, B, W=None):
    "int64 -> (sign in {-1,0,1}, digits[...,W] in [0,B-1]).  value = sign * sum_w digits[w]*B^w  (exact)."
    sign = torch.sign(n).to(torch.int64)
    a = n.abs()
    if W is None:
        W = digit_width(int(a.max().item()) if a.numel() else 0, B)
    digits = torch.empty(a.shape + (W,), dtype=torch.int64, device=n.device)
    tmp = a.clone()
    for w in range(W):
        digits[..., w] = tmp % B
        tmp = tmp // B
    return sign, digits, W

def recompose(sign, digits, B):
    "(sign,digits) -> exact int64 (positional notation, no rounding)."
    W = digits.shape[-1]
    powers = B ** torch.arange(W, dtype=torch.int64, device=digits.device)
    return sign * (digits * powers).sum(dim=-1)

def base_matmul(A_f, B_f, S, base):
    r"""float A(m,k) @ B(k,n), computed LOSSLESSLY through the base-B DIGIT PLANES.
    Ai[m,k] = sum_w sdA[m,k,w]*base^w  (sdA = signed digit in [-(base-1),base-1]).  So
      (Ai @ Bi)[m,n] = sum_{w,v} base^(w+v) * ( sdA[:,:,w] @ sdB[:,:,v] )
    -> digit-plane integer GEMMs weighted by positional powers; float multiply never appears. /S^2 undoes scaling.
    """
    Ai, Bi = to_fixed(A_f, S), to_fixed(B_f, S)
    sA, dA, WA = decompose(Ai, base)
    sB, dB, WB = decompose(Bi, base)
    sdA = (sA.unsqueeze(-1) * dA)          # (m,k,WA) signed digits
    sdB = (sB.unsqueeze(-1) * dB)          # (k,n,WB)
    m, n = A_f.shape[0], B_f.shape[1]
    P = torch.zeros((m, n), dtype=torch.int64, device=A_f.device)
    for w in range(WA):
        for v in range(WB):
            P += (base ** (w + v)) * (sdA[:, :, w] @ sdB[:, :, v])   # exact integer accumulation
    return P.to(torch.float64) / (S * S), (Ai, Bi)

# ================= VERIFICATION: 100% mathematical integrity, zero rounding in the decomposition =================
if __name__ == "__main__":
    torch.manual_seed(0)
    for base in (3, 5):                                   # ternary and quinary
        S = 10000
        A = torch.randn(6, 8, dtype=torch.float64)
        B = torch.randn(8, 5, dtype=torch.float64)
        print(f"\n===== base-{base} (S={S}) =====")

        # 1) DECOMPOSE + RECOMPOSE is EXACTLY lossless on the integers (== , not approx)
        Ai = to_fixed(A, S)
        s, d, W = decompose(Ai, base)
        Ar = recompose(s, d, base)
        print(f"  [1] decompose->recompose exact (int==int): {torch.equal(Ar, Ai)}   (digit width W={W})")

        # 2) base-native matmul EXACTLY equals the int64 fixed-point reference (zero loss vs plain int GEMM)
        out, (Ai2, Bi2) = base_matmul(A, B, S, base)
        ref_int = (Ai2 @ Bi2).to(torch.float64) / (S * S)
        print(f"  [2] base_matmul == int64 GEMM reference (allclose exact): {torch.allclose(out, ref_int, atol=0, rtol=0)}")

        # 3) matches true float matmul to the fixed-point precision (the ONLY floor, set by S -- a dial)
        true = A @ B
        maxerr = (out - true).abs().max().item()
        print(f"  [3] base_matmul ~ float A@B  (allclose, atol=3/S={3/S}): {torch.allclose(out, true, atol=3/S)}   maxerr={maxerr:.2e}")
    print("\n=> decomposition + base-native compute add ZERO error; precision floor = S (tunable). Not BitNet.")
