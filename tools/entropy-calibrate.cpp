// GPU entropisiert: Per-head entropy calibration tool for llama.cpp
//
// Calibrates a model's per-head attention entropy profile for use with
// entropy-adaptive KV cache compression (SCJedi method).
//
// Supports --cal-text <path> to use a longer text file for calibration
// sequences instead of the default short sentence starters.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-entropy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>
#include <sstream>

// Load a text file and split into n_segments roughly equal-length chunks,
// each capped at max_chars. Returns empty vector on failure.
static std::vector<std::string> load_cal_text(const std::string & path, int n_segments, size_t max_chars) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERR("Failed to open calibration text: %s\n", path.c_str());
        return {};
    }
    std::stringstream buf;
    buf << file.rdbuf();
    std::string text = buf.str();

    // Strip leading/trailing whitespace
    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);

    if (text.empty()) {
        LOG_ERR("Calibration text is empty\n");
        return {};
    }

    // If we want n_segments, split into roughly equal parts
    size_t total_len = std::min(text.size(), max_chars * n_segments);
    size_t seg_len = total_len / n_segments;

    std::vector<std::string> segments;
    size_t pos = 0;
    for (int i = 0; i < n_segments && pos < text.size(); ++i) {
        size_t end = std::min(pos + seg_len, text.size());
        // Try to break at sentence boundary near the end
        if (end < text.size()) {
            size_t search_end = std::min(end + 200, text.size());
            size_t sent_end = text.find_first_of(".!?\n", end);
            if (sent_end != std::string::npos && sent_end < search_end) {
                end = sent_end + 1;
            }
        }
        std::string seg = text.substr(pos, end - pos);
        // Trim whitespace
        seg.erase(0, seg.find_first_not_of(" \t\n\r"));
        seg.erase(seg.find_last_not_of(" \t\n\r") + 1);
        if (!seg.empty()) {
            segments.push_back(std::move(seg));
        }
        pos = end;
    }

    LOG_INF("Loaded %zu calibration segments from '%s'\n", segments.size(), path.c_str());
    return segments;
}

// Check for --cal-text flag before common_params_parse consumes all args
static std::string extract_cal_text_path(int & argc, char **& argv) {
    std::string path;
    int write_idx = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cal-text") == 0 && i + 1 < argc) {
            path = argv[i + 1];
            i++; // skip value
        } else {
            argv[write_idx++] = argv[i];
        }
    }
    argc = write_idx;
    argv[argc] = nullptr;
    return path;
}

int main(int argc, char ** argv) {
    std::string cal_text_path = extract_cal_text_path(argc, argv);

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, nullptr)) {
        return 1;
    }

    // Override for calibration: don't need huge context
    if (params.n_ctx == 0 || params.n_ctx > 2048) {
        params.n_ctx = 1024;
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

    // Build calibration prompts
    std::vector<const char *> prompts_c;
    std::vector<std::string> prompt_storage;

    if (!cal_text_path.empty()) {
        auto segments = load_cal_text(cal_text_path, 10, params.n_ctx * 4);  // ~4 chars per token
        if (segments.size() >= 3) {
            prompt_storage = std::move(segments);
            for (auto & s : prompt_storage) {
                // Truncate to roughly n_ctx tokens worth of chars
                if (s.size() > (size_t)params.n_ctx * 5) {
                    s.resize(params.n_ctx * 5);
                }
                prompts_c.push_back(s.c_str());
            }
        }
    }

    // Fallback to default short prompts if no text file loaded
    if (prompts_c.empty()) {
        LOG_INF("Using default calibration prompts (10 sentence starters, ~10 tokens each)\n");
        LOG_INF("For better results, provide longer text with --cal-text <file>\n");
        static const char * default_prompts[] = {
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
        for (auto * p : default_prompts) {
            prompts_c.push_back(p);
        }
    }

    auto * profile = llama_entropy_calibrate(ctx, prompts_c);
    if (profile) {
        profile->save("entropy_profile.json");
        llama_entropy_profile_free(profile);
        LOG_INF("Entropy profile saved to entropy_profile.json\n");
        llama_free(ctx);
        llama_model_free(model);
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

    srand(time(nullptr));
    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < n_heads; ++h) {
            llama_entropy_head eh;
            eh.layer = l;
            eh.head  = h;

            float entropy;
            float sink_weight;
            float r = (float)rand() / RAND_MAX;

            if (l < 3) {
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
