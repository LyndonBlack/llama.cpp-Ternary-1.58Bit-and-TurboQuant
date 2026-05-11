// GPU entropisiert: Per-head entropy calibration tool for llama.cpp
//
// Calibrates a model's per-head attention entropy profile for use with
// entropy-adaptive KV cache compression (SCJedi method).
//
// Currently generates a synthetic profile for framework testing.
// Real calibration requires attention-weight extraction from the graph.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-entropy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, nullptr)) {
        return 1;
    }

    // Override for calibration: shorter context, don't need long sequences
    if (params.n_ctx == 0 || params.n_ctx > 2048) {
        params.n_ctx = 512;
    }

    LOG_INF("Initializing model for entropy calibration...\n");

    // Initialize model from params
    auto llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx  = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("Failed to initialize model\n");
        return 1;
    }

    // Print model info
    int32_t n_layers = llama_model_n_layer(model);
    int32_t n_heads  = llama_model_n_head(model);
    int32_t n_embd   = llama_model_n_embd(model);

    LOG_INF("Model: %s\n", params.model.path.c_str());
    LOG_INF("  Layers: %d\n", n_layers);
    LOG_INF("  Heads:  %d\n", n_heads);
    LOG_INF("  Embed:  %d\n", n_embd);
    LOG_INF("  Ctx:    %d\n", params.n_ctx);

    // Try calibration (placeholder — returns null until hooks are implemented)
    std::vector<const char *> prompts = {
        "The theory of general relativity predicts that the universe is",
        "A neural network processes information by learning to recognize patterns in",
        "Machine learning algorithms can be used to solve complex problems in",
        "Natural language processing systems rely on transformer architectures to",
        "The process of evolution by natural selection results in organisms that are",
        "Climate change is driven by several factors including greenhouse gas emissions from",
        "The history of artificial intelligence spans several decades from early symbolic systems to",
        "Protein folding is a fundamental biological process where amino acid chains",
        "Renewable energy sources include solar and wind power which can",
        "The human brain contains approximately eighty-six billion neurons that communicate through",
    };

    auto * profile = llama_entropy_calibrate(ctx, prompts);
    if (profile) {
        profile->save("entropy_profile.json");
        llama_entropy_profile_free(profile);
        LOG_INF("Entropy profile saved to entropy_profile.json\n");
        return 0;
    }

    // Fallback: generate synthetic profile
    LOG_WRN("Full calibration not implemented in C++ yet.\n");
    LOG_WRN("Generating synthetic entropy profile for framework testing.\n");
    LOG_WRN("For real calibration, use:\n");
    LOG_WRN("  python3 tools/validate_entropy_adaptive.py --model <hf_name>\n");
    LOG_INF("\n");

    auto syn_profile = std::make_unique<llama_entropy_profile>();
    syn_profile->n_layers = n_layers;
    syn_profile->n_heads  = n_heads;

    float total_entropy = 0.0f;
    float min_ent = 999.0f, max_ent = 0.0f;

    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < n_heads; ++h) {
            llama_entropy_head eh;
            eh.layer = l;
            eh.head  = h;

            // Simulate head type distribution based on layer position
            float entropy;
            float sink_weight;
            float r = (float)rand() / RAND_MAX;

            if (l < 3) {
                // Early layers: more diffuse/mixed
                if (r < 0.50f) {
                    entropy = 0.2f + (float)rand() / RAND_MAX * 0.8f;
                    sink_weight = 0.6f + (float)rand() / RAND_MAX * 0.3f;
                } else if (r < 0.80f) {
                    entropy = 1.5f + (float)rand() / RAND_MAX * 1.0f;
                    sink_weight = 0.2f + (float)rand() / RAND_MAX * 0.2f;
                } else {
                    entropy = 2.5f + (float)rand() / RAND_MAX * 1.5f;
                    sink_weight = 0.05f + (float)rand() / RAND_MAX * 0.1f;
                }
            } else if (l < n_layers - 2) {
                // Middle layers: mostly sink/focused
                if (r < 0.70f) {
                    entropy = 0.1f + (float)rand() / RAND_MAX * 0.5f;
                    sink_weight = 0.7f + (float)rand() / RAND_MAX * 0.25f;
                } else if (r < 0.90f) {
                    entropy = 1.0f + (float)rand() / RAND_MAX * 1.5f;
                    sink_weight = 0.2f + (float)rand() / RAND_MAX * 0.3f;
                } else {
                    entropy = 2.5f + (float)rand() / RAND_MAX * 1.0f;
                    sink_weight = 0.1f + (float)rand() / RAND_MAX * 0.15f;
                }
            } else {
                // Late layers: mixed distribution
                if (r < 0.40f) {
                    entropy = 0.2f + (float)rand() / RAND_MAX * 1.0f;
                    sink_weight = 0.5f + (float)rand() / RAND_MAX * 0.3f;
                } else if (r < 0.80f) {
                    entropy = 1.5f + (float)rand() / RAND_MAX * 1.5f;
                    sink_weight = 0.15f + (float)rand() / RAND_MAX * 0.25f;
                } else {
                    entropy = 3.0f + (float)rand() / RAND_MAX * 1.5f;
                    sink_weight = 0.05f + (float)rand() / RAND_MAX * 0.1f;
                }
            }

            eh.entropy     = std::max(0.01f, entropy);
            eh.std_entropy = eh.entropy * 0.3f;
            eh.sink_weight = std::min(1.0f, std::max(0.0f, sink_weight));
            eh.n_entries   = std::max(1, (int)std::pow(2.0f, eh.entropy));

            total_entropy += eh.entropy;
            min_ent = std::min(min_ent, eh.entropy);
            max_ent = std::max(max_ent, eh.entropy);

            syn_profile->heads.push_back(eh);
        }
    }

    syn_profile->mean_entropy = total_entropy / (float)syn_profile->heads.size();

    float var = 0.0f;
    for (const auto & h : syn_profile->heads) {
        float d = h.entropy - syn_profile->mean_entropy;
        var += d * d;
    }
    syn_profile->std_entropy = std::sqrt(var / (float)syn_profile->heads.size());
    syn_profile->min_entropy = min_ent;
    syn_profile->max_entropy = max_ent;

    std::string out_path = "entropy_profile.json";
    syn_profile->save(out_path);

    LOG_INF("Synthetic profile saved to %s\n", out_path.c_str());
    LOG_INF("  Mean entropy: %.3f bits\n", syn_profile->mean_entropy);
    LOG_INF("  Range: %.3f - %.3f bits\n", syn_profile->min_entropy, syn_profile->max_entropy);
    LOG_INF("  CV: %.3f\n", syn_profile->std_entropy / std::max(0.001f, syn_profile->mean_entropy));

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
