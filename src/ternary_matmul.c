/* ternary_matmul.c -- lossless base-B integer arithmetic + a multiply-free ternary matmul.
 *
 * Framework-free. No GPU, no libraries beyond the C standard library, no floating-point multiply
 * in the hot path. Two exact self-checks:
 *   (1) writing an integer in base-3 or base-5 and reading it back is bit-exact (a change of base
 *       loses nothing -- this is NOT lossy quantization);
 *   (2) a multiply-free ternary matmul (weights in {-1,0,+1}) equals a plain integer GEMM exactly.
 *
 * Build & run:   cc -O2 -o ternary ternary_matmul.c && ./ternary
 *
 * Author: Jonathan David Wint.  License: MIT.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define W 20   /* base-B digit width -- ample for all values used below, no overflow */

/* ---- lossless base-B decomposition of an exact integer ----
 * value = sign * sum_{w=0}^{W-1} digit[w] * B^w ,  digits in [0, B-1]. */
static int decompose(int64_t n, int B, uint8_t *digit) {
    int sign = (n > 0) - (n < 0);
    int64_t a = llabs(n);
    for (int w = 0; w < W; w++) { digit[w] = (uint8_t)(a % B); a /= B; }
    return sign;
}
static int64_t reconstruct(int sign, const uint8_t *digit, int B) {
    int64_t a = 0, p = 1;
    for (int w = 0; w < W; w++) { a += (int64_t)digit[w] * p; p *= B; }
    return (int64_t)sign * a;
}

/* ---- fixed-point: float -> exact integer (scale S), no math.h needed ---- */
static int64_t to_fixed(double x, int64_t S) {
    double v = x * (double)S;
    return (int64_t)(v >= 0 ? v + 0.5 : v - 0.5);
}

/* ---- multiply-free ternary matmul: y[i] = sum_j Wt[i][j] * x[j], as add/skip/subtract ---- */
static void ternary_matmul(const int8_t *Wt, const int64_t *x, int64_t *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        int64_t acc = 0;
        const int8_t *row = Wt + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            int8_t w = row[j];
            if      (w > 0) acc += x[j];   /* +1 -> add       */
            else if (w < 0) acc -= x[j];   /* -1 -> subtract  */
            /* 0 -> skip.  No multiply appears anywhere. */
        }
        y[i] = acc;
    }
}
/* plain integer reference (real multiplies) to check the multiply-free version against */
static void ref_matmul(const int8_t *Wt, const int64_t *x, int64_t *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        int64_t acc = 0;
        for (int j = 0; j < N; j++) acc += (int64_t)Wt[(size_t)i * N + j] * x[j];
        y[i] = acc;
    }
}

int main(void) {
    int fails = 0;
    uint8_t d[W];

    /* check 1: base-3 and base-5 round-trip a wide range of integers, bit-exactly */
    for (int B = 3; B <= 5; B += 2) {
        int bad = 0;
        for (int64_t n = -100000; n <= 100000; n += 7) {
            int s = decompose(n, B, d);
            if (reconstruct(s, d, B) != n) { bad = 1; break; }
        }
        printf("base-%d  decompose -> reconstruct exact : %s\n", B, bad ? "FAIL" : "PASS");
        fails += bad;
    }

    /* also show it carries real (fixed-point) values with zero loss */
    {
        int64_t S = 10000; int B = 5, bad = 0;
        double xs[] = { 3.14159, -2.71828, 0.5, -0.0001, 42.0 };
        for (int k = 0; k < 5; k++) {
            int64_t fx = to_fixed(xs[k], S);
            int s = decompose(fx, B, d);
            if (reconstruct(s, d, B) != fx) bad = 1;
        }
        printf("base-5  fixed-point values round-trip   : %s\n", bad ? "FAIL" : "PASS");
        fails += bad;
    }

    /* check 2: multiply-free ternary matmul == integer reference, exactly */
    {
        int M = 8, N = 12, bad = 0;
        int8_t Wt[8 * 12];
        int64_t x[12], y[8], yref[8];
        srand(1);
        for (int i = 0; i < M * N; i++) Wt[i] = (int8_t)((rand() % 3) - 1);   /* {-1,0,+1} */
        for (int j = 0; j < N; j++)     x[j]  = (rand() % 20001) - 10000;
        ternary_matmul(Wt, x, y, M, N);
        ref_matmul(Wt, x, yref, M, N);
        for (int i = 0; i < M; i++) if (y[i] != yref[i]) bad = 1;
        printf("multiply-free matmul == integer GEMM    : %s\n", bad ? "FAIL" : "PASS");
        fails += bad;
    }

    printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASS -- lossless, and multiply-free.");
    return fails ? 1 : 0;
}
