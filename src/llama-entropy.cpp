#include "llama-entropy.h"

#include "llama.h"
#include "llama-context.h"
#include "llama-graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include <string>

//
// JSON helpers
//

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static std::string heads_to_json(const std::vector<llama_entropy_head> & heads) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < heads.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "{"
            << "\"layer\":" << heads[i].layer << ","
            << "\"head\":" << heads[i].head << ","
            << "\"entropy\":" << heads[i].entropy << ","
            << "\"std_entropy\":" << heads[i].std_entropy << ","
            << "\"sink_weight\":" << heads[i].sink_weight << ","
            << "\"n_entries\":" << heads[i].n_entries
            << "}";
    }
    oss << "]";
    return oss.str();
}

bool llama_entropy_profile::save(const std::string & path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to write entropy profile to %s\n", path.c_str());
        return false;
    }

    f << "{\n";
    f << "  \"n_layers\": " << n_layers << ",\n";
    f << "  \"n_heads\": " << n_heads << ",\n";
    f << "  \"mean_entropy\": " << mean_entropy << ",\n";
    f << "  \"min_entropy\": " << min_entropy << ",\n";
    f << "  \"max_entropy\": " << max_entropy << ",\n";
    f << "  \"std_entropy\": " << std_entropy << ",\n";
    f << "  \"heads\": " << heads_to_json(heads) << "\n";
    f << "}\n";

    f.close();
    fprintf(stderr, "Entropy profile saved to %s (%zu heads)\n", path.c_str(), heads.size());
    return true;
}

bool llama_entropy_profile::load(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to read entropy profile from %s\n", path.c_str());
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Helper: parse a scalar value by key (works on the top-level JSON object)
    auto read_val = [&](const std::string & key, bool is_float) -> double {
        std::string pat = "\"" + key + "\":";
        auto pos = content.find(pat);
        if (pos == std::string::npos) return 0;
        pos += pat.size();
        // Skip whitespace
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        // Read until comma, newline, or end of object
        size_t end = pos;
        while (end < content.size() && content[end] != ',' && content[end] != '\n' && content[end] != '\r' && content[end] != '}') end++;
        std::string val = content.substr(pos, end - pos);
        if (val.empty()) return 0;
        try {
            if (is_float) return std::stod(val);
            return (double)std::stoi(val);
        } catch (...) {
            fprintf(stderr, "Failed to parse %s='%s'\n", key.c_str(), val.c_str());
            return 0;
        }
    };

    n_layers = (int32_t)read_val("n_layers", false);
    n_heads  = (int32_t)read_val("n_heads", false);
    mean_entropy = (float)read_val("mean_entropy", true);
    min_entropy  = (float)read_val("min_entropy", true);
    max_entropy  = (float)read_val("max_entropy", true);
    std_entropy  = (float)read_val("std_entropy", true);

    // Find heads array
    auto hs = content.find("\"heads\":");
    if (hs == std::string::npos) { fprintf(stderr, "No heads array\n"); return false; }
    hs = content.find('[', hs);
    auto he = content.find(']', (hs != std::string::npos) ? hs : 0);
    if (hs == std::string::npos || he == std::string::npos) { fprintf(stderr, "No heads array\n"); return false; }

    std::string heads_str = content.substr(hs, he - hs + 1);
    heads.clear();

    size_t pos = 0;
    while ((pos = heads_str.find('{', pos)) != std::string::npos) {
        auto end = heads_str.find('}', pos);
        if (end == std::string::npos) break;
        std::string obj = heads_str.substr(pos, end - pos + 1);
        pos = end + 1;

        auto extract = [&](const std::string & key) -> std::string {
            std::string pat = "\"" + key + "\":";
            auto p = obj.find(pat);
            if (p == std::string::npos) return "";
            p += pat.size();
            auto e = obj.find_first_of(",}", p);
            return obj.substr(p, e - p);
        };

        llama_entropy_head h = {};
        auto parse_int = [&](const std::string & key) -> int {
            std::string s = extract(key);
            if (s.empty()) return 0;
            try { return std::stoi(s); } catch (...) { return 0; }
        };
        auto parse_float = [&](const std::string & key) -> float {
            std::string s = extract(key);
            if (s.empty()) return 0.0f;
            try { return std::stof(s); } catch (...) { return 0.0f; }
        };

        h.layer = parse_int("layer");
        h.head  = parse_int("head");
        h.entropy     = parse_float("entropy");
        h.std_entropy = parse_float("std_entropy");
        h.sink_weight = parse_float("sink_weight");
        h.n_entries   = parse_int("n_entries");
        heads.push_back(h);
    }

    fprintf(stderr, "Entropy profile loaded from %s (%zu heads)\n", path.c_str(), heads.size());
    return true;
}

//
// Entropy computation helpers
//

static float entropy_of_distribution(const float * data, int n, int stride) {
    double h = 0.0;
    for (int i = 0; i < n; ++i) {
        float p = data[i * stride];
        if (p > 1e-12f) {
            h += -p * log2f(p);
        }
    }
    return (float)h;
}

static float sink_weight_of_distribution(const float * data, int n, int stride) {
    if (n < 1) return 0.0f;
    return data[0]; // attention to position 0
}

static int classify_head_type(float entropy, float sink_w) {
    if (sink_w > 0.50f) return 0; // sink
    if (entropy < 1.5f) return 1; // focused
    if (entropy > 4.0f) return 2; // diffuse
    return 3;                     // mixed
}

static const char * head_type_name(int t) {
    static const char * names[] = {"sink", "focused", "diffuse", "mixed"};
    return (t >= 0 && t < 4) ? names[t] : "unknown";
}

//
// Main calibration function
//

struct llama_entropy_profile * llama_entropy_calibrate(
        struct llama_context * ctx,
        const std::vector<const char *> & prompts) {

    if (!ctx) {
        fprintf(stderr, "llama_entropy_calibrate: null context\n");
        return nullptr;
    }

    const int32_t n_layers = llama_model_n_layer(&ctx->get_model());
    const int32_t n_heads  = llama_model_n_head(&ctx->get_model());
    const int32_t n_ctx    = ctx->n_ctx();

    fprintf(stderr, "\n=== Entropy Calibration ===\n");
    fprintf(stderr, "Model: %d layers, %d heads/layer, %d total heads\n", n_layers, n_heads, n_layers * n_heads);
    fprintf(stderr, "Context: %d tokens\n", n_ctx);
    fprintf(stderr, "Calibration sequences: %zu\n", prompts.size());
    fprintf(stderr, "\n");

    // Save original FA state
    const bool orig_flash_attn = ctx->get_flash_attn();

    // Disable FlashAttention to get attention weights
    ctx->set_flash_attn(false);

    // Enable calibration output in the attention graph
    ctx->set_entropy_calibration(true);

    // Get model's tokenizer
    const llama_vocab * vocab = llama_model_get_vocab(&ctx->get_model());

    // Storage for per-head entropy accumulation
    const int32_t n_heads_total = n_layers * n_heads;
    struct Acc {
        double sum_entropy = 0.0;
        double sum_sink    = 0.0;
        int count = 0;
    };
    std::vector<Acc> accum(n_heads_total);

    // Note: We can't capture tensors during graph building (they're not allocated yet).
    // Instead, we iterate the executed graph AFTER compute to find softmax tensors.

    // Process each calibration prompt
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string prompt = prompts[pi];

        // Tokenize
        std::vector<llama_token> tokens(n_ctx + 4);
        int n_tok = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                    tokens.data(), (int)tokens.size(), false, false);
        if (n_tok <= 0) {
            fprintf(stderr, "  [%zu] tokenization failed for: %s\n", pi, prompt.c_str());
            continue;
        }
        if (n_tok > (int)n_ctx) {
            n_tok = (int)n_ctx;
        }
        tokens.resize(n_tok);

        // Create batch
        llama_batch batch = {
            /*n_tokens =*/ n_tok,
            /*token    =*/ tokens.data(),
            /*embd     =*/ nullptr,
            /*pos      =*/ nullptr,
            /*n_seq_id =*/ nullptr,
            /*seq_id   =*/ nullptr,
            /*logits   =*/ nullptr,
        };

        // Run inference
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "  [%zu] decode failed: %d\n", pi, ret);
            continue;
        }

        // Synchronize (ensure GPU data is available)
        ctx->synchronize();

        // Get the executed graph and find calib_out_N nodes by name
        ggml_cgraph * gf = ctx->get_last_graph();
        if (!gf) {
            fprintf(stderr, "  [%zu] no graph available\n", pi);
            continue;
        }

        int n_captured = 0;
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int ni = 0; ni < n_nodes; ++ni) {
            ggml_tensor * t = ggml_graph_node(gf, ni);
            if (!t || strncmp(t->name, "calib_out_", 10) != 0) {
                continue;
            }
            n_captured++;

            // Must have a buffer to read from
            if (!t->buffer) continue;

            // Get tensor dimensions
            const int64_t ne0 = t->ne[0]; // n_tokens_k
            const int64_t ne1 = t->ne[1]; // n_tokens_q
            const int64_t ne2 = t->ne[2]; // n_head
            const int64_t ne3 = t->ne[3]; // n_stream

            if (ne3 > 1) continue; // multi-stream not supported

            // Extract layer index from name "calib_out_N"
            int layer_idx = -1;
            if (sscanf(t->name, "calib_out_%d", &layer_idx) != 1 || layer_idx < 0) {
                continue;
            }

            // Read tensor data from GPU to host
            const size_t n_elems = (size_t)ne0 * ne1 * ne2 * ne3;
            std::vector<float> host_data(n_elems);
            ggml_backend_tensor_get(t, host_data.data(), 0, n_elems * sizeof(float));

            // Compute per-head entropy for each head in this layer
            for (int64_t ih = 0; ih < ne2; ++ih) {
                const int64_t head_offset = ih * ne0 * ne1;
                const int64_t global_head = layer_idx * n_heads + ih;
                if (global_head >= n_heads_total) continue;

                double sum_h = 0.0;
                double sum_s = 0.0;
                int    valid_q = 0;

                for (int64_t iq = 0; iq < ne1; ++iq) {
                    const int64_t n_keys = iq + 1; // causal
                    if (n_keys < 4) continue;

                    const float * attn_col = host_data.data() + head_offset + iq * ne0;

                    double h = 0.0;
                    for (int64_t ik = 0; ik < n_keys; ++ik) {
                        float p = attn_col[ik];
                        if (p > 1e-12f) {
                            h += -p * log2f(p);
                        }
                    }

                    sum_h += h;
                    sum_s += attn_col[0];
                    valid_q++;
                }

                if (valid_q > 0) {
                    accum[global_head].sum_entropy += sum_h / valid_q;
                    accum[global_head].sum_sink    += sum_s / valid_q;
                    accum[global_head].count++;
                }
            }
        }

        fprintf(stderr, "  [%zu] captured %d kq_soft_max tensors\n", pi, n_captured);
    }

    // Restore FA state
    ctx->set_flash_attn(orig_flash_attn);

    // Build the entropy profile
    auto * profile = new llama_entropy_profile();
    profile->n_layers = n_layers;
    profile->n_heads  = n_heads;

    double total_entropy = 0.0;
    double min_ent = 1e9, max_ent = 0.0;
    int valid_heads = 0;

    for (int il = 0; il < n_layers; ++il) {
        for (int ih = 0; ih < n_heads; ++ih) {
            const int idx = il * n_heads + ih;
            llama_entropy_head eh;
            eh.layer = il;
            eh.head  = ih;

            if (accum[idx].count > 0) {
                eh.entropy     = (float)(accum[idx].sum_entropy / accum[idx].count);
                eh.std_entropy = eh.entropy * 0.3f; // rough estimate
                eh.sink_weight = (float)(accum[idx].sum_sink / accum[idx].count);
                eh.n_entries   = std::max(1, (int)std::pow(2.0f, eh.entropy));

                total_entropy += eh.entropy;
                min_ent = std::min(min_ent, (double)eh.entropy);
                max_ent = std::max(max_ent, (double)eh.entropy);
                valid_heads++;
            } else {
                // Head not visited during calibration (shouldn't happen)
                eh.entropy     = 1.0f;
                eh.std_entropy = 0.3f;
                eh.sink_weight = 0.5f;
                eh.n_entries   = 2;
            }

            profile->heads.push_back(eh);
        }
    }

    if (valid_heads > 0) {
        profile->mean_entropy = (float)(total_entropy / valid_heads);
        double var = 0.0;
        for (const auto & h : profile->heads) {
            double d = h.entropy - profile->mean_entropy;
            var += d * d;
        }
        profile->std_entropy = (float)std::sqrt(var / profile->heads.size());
        profile->min_entropy = (float)min_ent;
        profile->max_entropy = (float)max_ent;

        fprintf(stderr, "\n=== Calibration Complete ===\n");
        fprintf(stderr, "Valid heads: %d/%d\n", valid_heads, (int)profile->heads.size());
        fprintf(stderr, "Mean entropy: %.3f bits\n", profile->mean_entropy);
        fprintf(stderr, "Entropy range: %.3f - %.3f bits\n", profile->min_entropy, profile->max_entropy);
        fprintf(stderr, "CV: %.3f\n", profile->std_entropy / std::max(0.001f, profile->mean_entropy));

        // Count head types
        int sink_count = 0, focused_count = 0, diffuse_count = 0, mixed_count = 0;
        for (const auto & h : profile->heads) {
            int t = classify_head_type(h.entropy, h.sink_weight);
            if (t == 0) sink_count++;
            else if (t == 1) focused_count++;
            else if (t == 2) diffuse_count++;
            else mixed_count++;
        }
        fprintf(stderr, "\nHead type distribution:\n");
        fprintf(stderr, "  sink:    %d (%.1f%%)\n", sink_count, 100.0f * sink_count / profile->heads.size());
        fprintf(stderr, "  focused: %d (%.1f%%)\n", focused_count, 100.0f * focused_count / profile->heads.size());
        fprintf(stderr, "  diffuse: %d (%.1f%%)\n", diffuse_count, 100.0f * diffuse_count / profile->heads.size());
        fprintf(stderr, "  mixed:   %d (%.1f%%)\n", mixed_count, 100.0f * mixed_count / profile->heads.size());
    }

    return profile;
}

void llama_entropy_profile_free(struct llama_entropy_profile * profile) {
    delete profile;
}

//
// Budget computation
//

std::vector<float> llama_entropy_compute_budgets(
        const struct llama_entropy_profile & profile,
        float compression_ratio) {

    std::vector<float> budgets;
    budgets.reserve(profile.heads.size());

    double r = 1.0 / compression_ratio;
    double mean_ent = profile.mean_entropy;

    if (mean_ent <= 0.0f) {
        for (size_t i = 0; i < profile.heads.size(); ++i) {
            budgets.push_back((float)r);
        }
        return budgets;
    }

    for (const auto & h : profile.heads) {
        double scale = h.entropy / mean_ent;
        scale = std::max(0.3, std::min(2.5, scale));
        double budget = r * scale;
        budget = std::min(1.0, budget);
        budgets.push_back((float)budget);
    }

    return budgets;
}

std::vector<std::vector<bool>> llama_entropy_apply_budgets(
        const struct llama_entropy_profile & /*profile*/,
        const std::vector<float> & /*budgets*/,
        int32_t /*n_ctx*/) {
    return {}; // placeholder
}
