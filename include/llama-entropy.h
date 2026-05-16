#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <cstdint>

//
// Per-head entropy profile for entropy-adaptive KV cache compression.
//
// Inspired by SCJedi/entropy-adaptive-kv-cache:
//   Per-head entropy-based KV cache budget allocation.
//   Low-entropy (sink/focused) heads get fewer KV entries;
//   high-entropy (diffuse/mixed) heads retain full context.
//

// Per-head entropy record
struct llama_entropy_head {
    int32_t layer;
    int32_t head;
    float   entropy;       // mean attention entropy in bits
    float   std_entropy;   // standard deviation across query positions
    float   sink_weight;   // fraction of attention on position 0
    int32_t n_entries;     // estimated KV entries needed (2^entropy)
};

// Full entropy profile for a model
struct LLAMA_API llama_entropy_profile {
    int32_t                   n_layers = 0;
    int32_t                   n_heads  = 0;
    float                     mean_entropy = 0.0f;
    float                     min_entropy  = 0.0f;
    float                     max_entropy  = 0.0f;
    float                     std_entropy  = 0.0f;
    std::vector<llama_entropy_head> heads;

    // Serialize to JSON
    bool save(const std::string & path);
    bool load(const std::string & path);
};

// Calibration function: run the model on calibration sequences and
// capture per-head attention entropy.
// Returns the entropy profile, or nullptr on failure.
LLAMA_API struct llama_entropy_profile * llama_entropy_calibrate(
        struct llama_context * ctx,
        const std::vector<const char *> & prompts);  // calibration sequences

LLAMA_API void llama_entropy_profile_free(struct llama_entropy_profile * profile);

// Given an entropy profile and a target compression ratio, compute
// per-head retention budgets.
// Returns a vector of per-head retention ratios in [0, 1].
std::vector<float> llama_entropy_compute_budgets(
        const struct llama_entropy_profile & profile,
        float compression_ratio);  // e.g., 2.0 for 2x compression

// Convert entropy-based budget to a per-layer eviction mask.
// budget[h] = retention ratio for head h
// Returns a vector<bool> of length n_ctx for each head indicating which tokens to keep.
// (This is a placeholder for future implementation.)
std::vector<std::vector<bool>> llama_entropy_apply_budgets(
        const struct llama_entropy_profile & profile,
        const std::vector<float> & budgets,
        int32_t n_ctx);
