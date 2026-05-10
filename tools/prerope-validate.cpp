// Pre-RoPE Turbo4 quantization error validation
// gcc -c -O2 -o /tmp/turbo-quant.o ggml/src/ggml-turbo-quant.c \
//   -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu
// g++ -std=c++17 -O2 -o build/bin/prerope-validate tools/prerope-validate.cpp \
//   /tmp/turbo-quant.o -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -lm

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>

// Minimal types needed
typedef uint16_t ggml_half;
#define QK_TURBO4 128

struct block_turbo4_0 {
    ggml_half  norm;
    ggml_half  rnorm;
    uint8_t    qs[QK_TURBO4 / 2]; // 64 bytes, 128 x 4 bits
};
static_assert(sizeof(block_turbo4_0) == 68, "turbo4 block size");

extern "C" {
void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k);
void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k);
}

static void apply_rope(float * v, int nr, int pos, float fb, float fs) {
    for (int i = 0; i < nr; i += 2) {
        float th = pos * powf(fb, -2.0f * i / nr) * fs;
        float c = cosf(th), s = sinf(th);
        float v0 = v[i], v1 = v[i+1];
        v[i] = v0*c - v1*s; v[i+1] = v0*s + v1*c;
    }
}

static double snr(const float * ref, const float * tst, int n) {
    double sig = 0, noi = 0;
    for (int i = 0; i < n; ++i) {
        sig += ref[i] * (double)ref[i];
        double e = ref[i] - (double)tst[i];
        noi += e * e;
    }
    return noi < 1e-30 ? 200.0 : 10.0 * log10(sig / noi);
}

int main() {
    const int nr = 128, nt = 24;
    const float fb = 1000000.0f, fs = 0.25f;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1);

    const int N = nr * nt;
    std::vector<float> raw(N), pre_rope(N), post_deq(N);
    std::vector<block_turbo4_0> buf((nr + QK_TURBO4 - 1) / QK_TURBO4);

    for (int i = 0; i < N; ++i) raw[i] = nd(rng);

    printf("=== Pre-RoPE vs Post-RoPE Turbo4 Quantization ===\n");
    printf("Head dim %d, %d tokens, freq_base=%.0f, freq_scale=%.2f\n\n", nr, nt, fb, fs);

    double sum_pre = 0, sum_post = 0;

    for (int t = 0; t < nt; ++t) {
        const float * kr = raw.data() + t * nr;

        // Pre-RoPE: quantize(K_raw) → dequant → RoPE
        float * pr = pre_rope.data() + t * nr;
        quantize_row_turbo4_0_ref(kr, buf.data(), nr);
        dequantize_row_turbo4_0(buf.data(), pr, nr);
        apply_rope(pr, nr, t, fb, fs);

        // Post-RoPE: RoPE(K_raw) → quantize → dequant
        float * pd = post_deq.data() + t * nr;
        memcpy(pd, kr, nr * 4);
        apply_rope(pd, nr, t, fb, fs);
        quantize_row_turbo4_0_ref(pd, buf.data(), nr);
        dequantize_row_turbo4_0(buf.data(), pd, nr);

        // Reference: RoPE without quantization
        float ref[128];
        memcpy(ref, kr, nr * 4);
        apply_rope(ref, nr, t, fb, fs);

        double sp = snr(ref, pr, nr);
        double so = snr(ref, pd, nr);
        sum_pre += sp; sum_post += so;
        printf("t=%2d  pre-rope %7.2f dB  post-rope %7.2f dB  Δ %+.2f dB\n",
               t, sp, so, sp - so);
    }

    printf("\nAVG  pre-rope %.2f dB  post-rope %.2f dB  Δ %+.2f dB\n",
           sum_pre/nt, sum_post/nt, (sum_pre-sum_post)/nt);
    return 0;
}
