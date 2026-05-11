// RVQ (Residual Vector Quantization) vs flat TurboQuant — SNR comparison
// Tests whether 2+2-bit RVQ stages beat 4-bit turbo4 at the same memory.
//
// Build: gcc -c -O2 -o /tmp/turbo-quant.o ggml/src/ggml-turbo-quant.c \
//          -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu
//        g++ -std=c++17 -O2 -o build/bin/rvq-validate tools/rvq-validate.cpp \
//          /tmp/turbo-quant.o -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -lm
//
// Usage: ./build/bin/rvq-validate [--tokens N] [--seed S] [--dim-var]

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <cstdlib>

typedef uint16_t ggml_half;

// --- Import turbo2 block types ---
#define QK_TURBO2 128
struct block_turbo2_0 {
    ggml_half  norm;
    uint8_t    qs[QK_TURBO2 / 4];
};
extern "C" {
void quantize_row_turbo2_0_ref(const float * x, block_turbo2_0 * y, int64_t k);
void dequantize_row_turbo2_0(const block_turbo2_0 * x, float * y, int64_t k);
}

// --- Import turbo4 block types ---
#define QK_TURBO4 128
struct block_turbo4_0 {
    ggml_half  norm;
    ggml_half  rnorm;
    uint8_t    qs[QK_TURBO4 / 2];
};
extern "C" {
void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
}

static double snr_db(const float * ref, const float * tst, int n) {
    double sig = 0, noi = 0;
    for (int i = 0; i < n; ++i) {
        sig += ref[i] * (double)ref[i];
        double e = ref[i] - (double)tst[i];
        noi += e * e;
    }
    return (noi < 1e-30) ? 200.0 : 10.0 * log10(sig / noi);
}

int main(int argc, char ** argv) {
    int n_rot = 128, n_tokens = 24, seed = 42;
    bool dim_var = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--tokens") && i+1 < argc) n_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim-var")) dim_var = true;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: rvq-validate [options]\n");
            printf("  --tokens N    Number of tokens (default: 24)\n");
            printf("  --seed N      Random seed (default: 42)\n");
            printf("  --dim-var     Dimension-dependent variance\n");
            return 0;
        }
    }

    // Generate K values
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> raw(n_rot * n_tokens);
    for (int i = 0; i < n_rot * n_tokens; ++i) {
        raw[i] = nd(rng);
        if (dim_var) {
            int d = i % n_rot;
            raw[i] *= expf(-d / 32.0f);
        }
    }

    const int nb = (n_rot + QK_TURBO4 - 1) / QK_TURBO4;
    std::vector<block_turbo4_0> buf4(nb);
    std::vector<block_turbo2_0> buf2((n_rot + QK_TURBO2 - 1) / QK_TURBO2);
    std::vector<float> tmp(n_rot);
    std::vector<float> residual(n_rot);

    double avg_turbo4 = 0, avg_rvq_2plus2 = 0, avg_turbo6_approx = 0;

    printf("=== RVQ vs TurboQuant SNR Comparison ===\n");
    printf("Head dim %d, %d tokens, dim-var=%s\n\n", n_rot, n_tokens,
           dim_var ? "yes" : "no");
    printf("Comparing at equal bits per value:\n");
    printf("  turbo4:       4 bits (1 stage, 16 centroids)\n");
    printf("  rvq 2+2:      4 bits (2 stages, 4+4 centroids)\n");
    printf("  turbo6 (sim): 6 bits (64 centroids)\n\n");

    for (int t = 0; t < n_tokens; ++t) {
        const float * kr = raw.data() + t * n_rot;

        // Reference (no quantization)
        float ref[128];
        memcpy(ref, kr, n_rot * 4);

        // --- Turbo4 (flat 4-bit) ---
        float turbo4_deq[128];
        quantize_row_turbo4_0_ref(kr, buf4.data(), n_rot);
        dequantize_row_turbo4_0(buf4.data(), turbo4_deq, n_rot);
        double s_turbo4 = snr_db(ref, turbo4_deq, n_rot);

        // --- RVQ 2+2 (2-bit x 2 stages, same memory as 4-bit) ---
        float rvq_rec[128];
        memset(rvq_rec, 0, n_rot * 4);
        
        // Stage 0: 2-bit quantize on original
        quantize_row_turbo2_0_ref(kr, buf2.data(), n_rot);
        dequantize_row_turbo2_0(buf2.data(), tmp.data(), n_rot);
        for (int i = 0; i < n_rot; ++i) rvq_rec[i] = tmp[i];
        
        // Residual: original - stage0 reconstruction
        for (int i = 0; i < n_rot; ++i) residual[i] = kr[i] - tmp[i];
        
        // Stage 1: 2-bit quantize on residual
        quantize_row_turbo2_0_ref(residual.data(), buf2.data(), n_rot);
        dequantize_row_turbo2_0(buf2.data(), tmp.data(), n_rot);
        for (int i = 0; i < n_rot; ++i) rvq_rec[i] += tmp[i];
        
        double s_rvq = snr_db(ref, rvq_rec, n_rot);

        // --- turbo6 approximation (6-bit via turbo4+scaled residual) ---
        // Better approximation: quantize residual of turbo4 with another turbo4
        // But with fewer bits = use turbo2 on scaled residual
        float turbo6_approx[128];
        float scale_factor = 0.5f; // residual typically smaller
        quantize_row_turbo4_0_ref(kr, buf4.data(), n_rot);
        dequantize_row_turbo4_0(buf4.data(), turbo6_approx, n_rot);
        for (int i = 0; i < n_rot; ++i) residual[i] = (kr[i] - turbo6_approx[i]) / scale_factor;
        quantize_row_turbo4_0_ref(residual.data(), buf4.data(), n_rot);
        dequantize_row_turbo4_0(buf4.data(), tmp.data(), n_rot);
        for (int i = 0; i < n_rot; ++i) turbo6_approx[i] += tmp[i] * scale_factor;
        double s_turbo6 = snr_db(ref, turbo6_approx, n_rot);

        avg_turbo4 += s_turbo4;
        avg_rvq_2plus2 += s_rvq;
        avg_turbo6_approx += s_turbo6;

        if (n_tokens <= 24 || t < 5 || t % (n_tokens/4) == 0 || t == n_tokens-1) {
            printf("t=%4d  turbo4 %7.2f dB  rvq2+2 %7.2f dB  Δ %+.2f dB  |  turbo6~ %7.2f dB\n",
                   t, s_turbo4, s_rvq, s_rvq - s_turbo4, s_turbo6);
        }
    }

    printf("\n=== AVERAGES (%d tokens) ===\n", n_tokens);
    printf("turbo4 (4-bit):       %.2f dB\n", avg_turbo4 / n_tokens);
    printf("rvq 2+2 (4-bit):      %.2f dB  Δ vs turbo4: %+.2f dB\n",
           avg_rvq_2plus2 / n_tokens, (avg_rvq_2plus2 - avg_turbo4) / n_tokens);
    printf("turbo6~ (approx):     %.2f dB\n", avg_turbo6_approx / n_tokens);

    printf("\n=== MEMORY COMPARISON (per head per token) ===\n");
    printf("turbo4:   %d bytes = %.1f bits/value\n", nb * 68, 4.0);
    printf("rvq 2+2:  %d bytes = %.1f bits/value  (2 turbo2 stages)\n",
           2 * (int)buf2.size() * (int)sizeof(block_turbo2_0),
           2 * 2.0);
    printf("turbo6:   %d bytes = %.1f bits/value\n", nb * 84, 6.0);
    // turbo4 size = 68 bytes / 128 values * 8 = 4.25 bits/val
    // turbo2 size = 34 bytes / 128 values * 8 = 2.125 bits/val, 2 stages = 4.25 bits/val
    // turbo6 size = 84 bytes / 128 values * 8 = 5.25 bits/val

    return 0;
}
