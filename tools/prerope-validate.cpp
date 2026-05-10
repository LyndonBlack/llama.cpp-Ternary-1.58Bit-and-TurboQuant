// Pre-RoPE Turbo4 Quantization Validation — parameterized
// Tests whether pre-RoPE quantization introduces less error than post-RoPE
// using Qwen3.6-compatible parameters and optionally real K values from file.
//
// Build: gcc -c -O2 -o /tmp/turbo-quant.o ggml/src/ggml-turbo-quant.c \
//          -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu
//        g++ -std=c++17 -O2 -o build/bin/prerope-validate tools/prerope-validate.cpp \
//          /tmp/turbo-quant.o -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -lm
//
// Usage:
//   ./prerope-validate                          # Random Gaussian, Qwen params
//   ./prerope-validate --freq-base 10000        # Standard RoPE
//   ./prerope-validate --tokens 128             # Longer context
//   ./prerope-validate --from-file /tmp/k.bin   # Real K values from model
//   ./prerope-validate --freq-base 10000 --freq-scale 1.0 --seed 123

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <cstdlib>

typedef uint16_t ggml_half;
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

static void apply_rope(float * v, int nr, int pos, float fb, float fs) {
    for (int i = 0; i < nr; i += 2) {
        float th = pos * powf(fb, -2.0f * i / nr) * fs;
        float c = cosf(th), s = sinf(th);
        float v0 = v[i], v1 = v[i+1];
        v[i] = v0*c - v1*s; v[i+1] = v0*s + v1*c;
    }
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
    // Defaults — Qwen3.6
    int n_rot = 128, n_tokens = 24, seed = 42;
    float freq_base = 1000000.0f, freq_scale = 0.25f;
    const char * from_file = nullptr;
    bool dim_var = false; // dimension-dependent variance (more realistic)

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--freq-base") && i+1 < argc)
            freq_base = atof(argv[++i]);
        else if (!strcmp(argv[i], "--freq-scale") && i+1 < argc)
            freq_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc)
            n_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--from-file") && i+1 < argc)
            from_file = argv[++i];
        else if (!strcmp(argv[i], "--dim-var"))
            dim_var = true;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: prerope-validate [options]\n");
            printf("  --freq-base F    RoPE frequency base (default: 1000000)\n");
            printf("  --freq-scale S   RoPE frequency scale (default: 0.25)\n");
            printf("  --tokens N       Number of tokens (default: 24)\n");
            printf("  --seed N         Random seed (default: 42)\n");
            printf("  --from-file F    Read K values from binary file (overrides random)\n");
            printf("  --dim-var        Use dimension-dependent variance (more realistic)\n");
            return 0;
        }
    }

    std::vector<float> raw;
    int n_dim = n_rot;

    if (from_file) {
        // Read K values from binary file
        FILE * fp = fopen(from_file, "rb");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", from_file); return 1; }
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        rewind(fp);
        n_tokens = fsize / (n_dim * 4);
        if (n_tokens == 0) { fprintf(stderr, "Empty file\n"); return 1; }
        raw.resize(n_tokens * n_dim);
        fread(raw.data(), 4, n_tokens * n_dim, fp);
        fclose(fp);
        printf("Read %d tokens x %d dims from %s\n", n_tokens, n_dim, from_file);
    } else {
        // Generate synthetic K values
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0, 1);
        raw.resize(n_tokens * n_dim);
        for (int i = 0; i < n_tokens * n_dim; ++i) {
            raw[i] = nd(rng);
            if (dim_var) {
                // Dimension-dependent variance: lower dims have more variance
                int d = i % n_dim;
                raw[i] *= expf(-d / 32.0f);  // ~5% at dim 96
            }
        }
    }

    const int nb = (n_dim + QK_TURBO4 - 1) / QK_TURBO4;
    std::vector<block_turbo4_0> buf(nb);
    std::vector<float> pre_rope(n_tokens * n_dim);
    std::vector<float> post_deq(n_tokens * n_dim);

    printf("=== Pre-RoPE vs Post-RoPE Turbo4 Quantization ===\n");
    printf("Head dim %d, %d tokens, freq_base=%.0f, freq_scale=%.2f\n",
           n_dim, n_tokens, freq_base, freq_scale);
    if (dim_var) printf("  dimension-dependent variance (exp(-d/32))\n");
    if (from_file) printf("  real K values from: %s\n", from_file);

    double sum_pre = 0, sum_post = 0;
    int n_pos = 0;

    for (int t = 0; t < n_tokens; ++t) {
        const float * kr = raw.data() + t * n_dim;

        // Pre-RoPE: quantize(K_raw) → dequant → RoPE
        float * pr = pre_rope.data() + t * n_dim;
        quantize_row_turbo4_0_ref(kr, buf.data(), n_dim);
        dequantize_row_turbo4_0(buf.data(), pr, n_dim);
        apply_rope(pr, n_dim, t, freq_base, freq_scale);

        // Post-RoPE: RoPE(K_raw) → quantize → dequant
        float * pd = post_deq.data() + t * n_dim;
        memcpy(pd, kr, n_dim * 4);
        apply_rope(pd, n_dim, t, freq_base, freq_scale);
        quantize_row_turbo4_0_ref(pd, buf.data(), n_dim);
        dequantize_row_turbo4_0(buf.data(), pd, n_dim);

        // Reference: RoPE without quantization
        float ref[256];
        memcpy(ref, kr, n_dim * 4);
        apply_rope(ref, n_dim, t, freq_base, freq_scale);

        double sp = snr_db(ref, pr, n_dim);
        double so = snr_db(ref, pd, n_dim);
        sum_pre += sp; sum_post += so;
        n_pos++;
        if (n_tokens <= 24 || t < 5 || t % (n_tokens/4) == 0 || t == n_tokens-1) {
            printf("t=%4d  pre %7.2f dB  post %7.2f dB  Δ %+.2f dB\n",
                   t, sp, so, sp - so);
        }
    }

    printf("\nAVG  pre-rope %.2f dB  post-rope %.2f dB  Δ %+.2f dB  (%d tokens)\n",
           sum_pre/n_pos, sum_post/n_pos, (sum_pre-sum_post)/n_pos, n_pos);
    return 0;
}
