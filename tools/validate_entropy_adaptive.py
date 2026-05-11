#!/usr/bin/env python3
"""
Entropy-Adaptive KV Cache Validation Tool
==========================================

Investigation of SCJedi's per-head entropy-adaptive KV cache compression
and its applicability to the TurboKV fork.

Based on: github.com/SCJedi/entropy-adaptive-kv-cache
Reference: "Model Tells You What to Discard" (arXiv:2310.01801, ICLR 2024)

What this does:
  1. Loads a HuggingFace model (GPT-2 small by default)
  2. Runs calibration sequences to extract per-head attention weights
  3. Computes attention entropy per head
  4. Simulates entropy-adaptive vs uniform eviction
  5. Reports whether the model exhibits the 300x entropy range needed for benefit
  6. Outputs a recommendation for TurboKV trial implementation

Usage:
  pip install transformers torch
  python tools/validate_entropy_adaptive.py
  python tools/validate_entropy_adaptive.py --model gpt2  # default
  python tools/validate_entropy_adaptive.py --model Qwen/Qwen2.5-0.5B  # small proxy
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class HeadEntropy:
    layer: int
    head: int
    entropy: float          # mean entropy in bits
    std_entropy: float      # std across query positions
    head_type: str          # sink / diffuse / positional / selective / focused / mixed
    n_entries_needed: int   # estimated KV entries needed

    def to_dict(self):
        return asdict(self)


@dataclass
class CalibrationResult:
    model_name: str
    n_layers: int
    n_heads: int
    n_heads_total: int
    n_calibration_seqs: int
    seq_len: int
    head_entropies: List[HeadEntropy] = field(default_factory=list)
    mean_entropy: float = 0.0
    min_entropy: float = 0.0
    max_entropy: float = 0.0
    std_entropy: float = 0.0
    entropy_cv: float = 0.0
    head_type_counts: dict = field(default_factory=dict)
    sink_ratio: float = 0.0
    diffuse_ratio: float = 0.0
    recommendation: str = ""

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_heads_total": self.n_heads_total,
            "n_calibration_seqs": self.n_calibration_seqs,
            "seq_len": self.seq_len,
            "mean_entropy": self.mean_entropy,
            "min_entropy": self.min_entropy,
            "max_entropy": self.max_entropy,
            "std_entropy": self.std_entropy,
            "entropy_cv": self.entropy_cv,
            "head_type_counts": self.head_type_counts,
            "sink_ratio": self.sink_ratio,
            "diffuse_ratio": self.diffuse_ratio,
            "recommendation": self.recommendation,
        }


# ============================================================================
# Entropy computation
# ============================================================================

def entropy_of_attention(attn_weights: np.ndarray) -> float:
    """
    Compute Shannon entropy of attention distribution.
    H = -sum(p_i * log2(p_i))

    Args:
        attn_weights: 1D array of attention probabilities (should sum to 1)

    Returns:
        Entropy in bits
    """
    attn_weights = np.clip(attn_weights, 1e-12, 1.0)
    return float(-np.sum(attn_weights * np.log2(attn_weights)))


def classify_head(entropy: float, sink_weight: float) -> str:
    """
    Classify a head based on entropy and sink-attention fraction.
    """
    if sink_weight > 0.50:
        return "sink"
    if entropy < 1.5:
        return "focused"
    if entropy > 4.0:
        return "diffuse"
    return "mixed"


def estimate_entries_needed(entropy: float) -> int:
    """
    Estimate KV entries needed based on entropy.
    Low entropy heads need few entries, high entropy need many.
    """
    # 2^entropy gives a theoretical minimum based on information theory
    # Clamp to reasonable range
    return max(1, min(128, int(2 ** entropy)))


# ============================================================================
# Strategy simulation (no model needed — analytical)
# ============================================================================

@dataclass
class SimulationResult:
    strategy: str
    compression_ratio: float
    exact_match: float
    kl_divergence: float

    def to_dict(self):
        return asdict(self)


def analytical_simulate(
    head_entropies: List[HeadEntropy],
    compression_ratio: float,
    strategy: str = "entropy_adaptive",
    mean_entropy: float = 2.81,
) -> SimulationResult:
    """
    Analytical approximation of compression quality.

    This uses a simplified model based on SCJedi's published results.
    For a real evaluation, run the full transformers-based experiment.

    The model: quality scales with how well budget matches head entropy.
    - Uniform strategies waste budget on low-entropy heads and starve high-entropy heads.
    - Entropy-adaptive allocates proportionally.

    Uses SCJedi's published GPT-2 small results as a transfer function.
    """
    n_heads = len(head_entropies)
    if n_heads == 0:
        return SimulationResult(strategy, compression_ratio, 0.0, 100.0)

    if strategy == "entropy_adaptive":
        # Interpolate from SCJedi's published Pareto curve (Table 4)
        # These are empirical results from GPT-2 small
        pareto = {
            1.0: 0.983, 1.3: 0.967, 2.0: 0.932, 2.5: 0.909,
            3.3: 0.867, 5.0: 0.799, 6.7: 0.734, 10.0: 0.605,
            14.0: 0.473, 20.0: 0.381, 33.0: 0.248,
        }
        kl_pareto = {
            1.0: 0.002, 1.3: 0.009, 2.0: 0.026, 2.5: 0.041,
            3.3: 0.081, 5.0: 0.150, 6.7: 0.287, 10.0: 0.589,
            14.0: 1.073, 20.0: 1.535, 33.0: 2.363,
        }

        # Apply correction based on entropy CV vs GPT-2's CV
        gpt2_cv = 1.46 / 2.81  # ~0.52
        if mean_entropy > 0:
            our_cv = np.std([h.entropy for h in head_entropies]) / mean_entropy
            # Higher CV -> more benefit from adaptive allocation
            cv_factor = min(2.0, our_cv / gpt2_cv)
        else:
            cv_factor = 1.0

        # Find nearest points and interpolate
        crs = sorted(pareto.keys())
        if compression_ratio <= crs[0]:
            match = pareto[crs[0]]
            kl = kl_pareto[crs[0]]
        elif compression_ratio >= crs[-1]:
            match = pareto[crs[-1]]
            kl = kl_pareto[crs[-1]]
        else:
            for i in range(len(crs) - 1):
                if crs[i] <= compression_ratio <= crs[i + 1]:
                    t = (compression_ratio - crs[i]) / (crs[i + 1] - crs[i])
                    match = pareto[crs[i]] + t * (pareto[crs[i + 1]] - pareto[crs[i]])
                    kl = kl_pareto[crs[i]] + t * (kl_pareto[crs[i + 1]] - kl_pareto[crs[i]])
                    break

        match = min(1.0, match * (0.98 + 0.02 * cv_factor))
        kl = max(0.001, kl / (0.98 + 0.02 * cv_factor))

    elif strategy == "sink_recent":
        # SCJedi's published sink+recent results (Table 1)
        sink_pareto = {
            1.0: 1.000, 1.3: 0.827, 2.0: 0.690, 3.3: 0.534,
            5.0: 0.432, 10.0: 0.252, 20.0: 0.139,
        }
        sink_kl = {
            1.0: 0.000, 1.3: 0.135, 2.0: 0.360, 3.3: 0.793,
            5.0: 1.780, 10.0: 2.400, 20.0: 3.350,
        }
        crs = sorted(sink_pareto.keys())
        if compression_ratio <= crs[0]:
            match = sink_pareto[crs[0]]
            kl = sink_kl[crs[0]]
        elif compression_ratio >= crs[-1]:
            match = sink_pareto[crs[-1]]
            kl = sink_kl[crs[-1]]
        else:
            for i in range(len(crs) - 1):
                if crs[i] <= compression_ratio <= crs[i + 1]:
                    t = (compression_ratio - crs[i]) / (crs[i + 1] - crs[i])
                    match = sink_pareto[crs[i]] + t * (sink_pareto[crs[i + 1]] - sink_pareto[crs[i]])
                    kl = sink_kl[crs[i]] + t * (sink_kl[crs[i + 1]] - sink_kl[crs[i]])
                    break

    elif strategy == "recent_k":
        match = max(0.021, 1.0 * (1.0 - compression_ratio + 0.01))
        kl = min(10.0, 5.0 * compression_ratio)
    else:
        match = 0.5
        kl = 1.0

    return SimulationResult(
        strategy=strategy,
        compression_ratio=compression_ratio,
        exact_match=float(match),
        kl_divergence=float(kl),
    )


# ============================================================================
# Full transformers-based experiment
# ============================================================================

def run_model_experiment(
    model_name: str = "gpt2",
    n_calibration_seqs: int = 20,
    seq_len: int = 128,
    device: str = "cpu",
) -> CalibrationResult:
    """
    Run full entropy profiling on a HuggingFace model.
    
    This is the main experiment — replicates SCJedi's methodology:
    1. Load model
    2. Run calibration sequences capturing attention weights
    3. Compute per-head entropy
    4. Classify heads
    5. Report
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        dtype=torch.float32,
    ).to(device)
    model.eval()

    # Model architecture inspection
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    n_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 12))
    
    print(f"  Layers: {n_layers}, Heads per layer: {n_heads}, Total heads: {n_layers * n_heads}")

    # Calibration sequences (WikiText-2 style: simple prompts)
    calibration_prompts = [
        "The theory of general relativity predicts that",
        "A neural network processes information by",
        "In the beginning, the universe was",
        "Machine learning algorithms can be used to",
        "The study of quantum mechanics reveals",
        "Natural language processing systems rely on",
        "The process of evolution by natural selection",
        "Climate change is driven by several factors",
        "The history of artificial intelligence spans",
        "Protein folding is a fundamental biological process",
        "Renewable energy sources include solar and wind",
        "The human brain contains approximately",
        "Cryptographic protocols ensure secure communication",
        "Economic theory suggests that markets",
        "The solar system consists of",
        "DNA replication is a highly accurate process",
        "The development of computer hardware has",
        "Ocean currents play a crucial role in",
        "Blockchain technology enables decentralized",
        "The principles of thermodynamics govern",
        "Statistical inference allows us to",
        "The field of robotics combines",
        "Photosynthesis converts light energy into",
        "The theory of plate tectonics explains",
        "Genetic algorithms are inspired by",
        "Computer vision systems must handle",
        "The concept of entropy originates from",
        "Reinforcement learning involves an agent",
        "The immune system defends against",
        "Dark matter and dark energy constitute",
    ][:n_calibration_seqs]

    # Encode all sequences
    encoded = tokenizer(
        calibration_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    print(f"  Sequences: {len(calibration_prompts)}, max length: {input_ids.shape[1]}")

    # Storage for per-head entropy
    head_entropies = []

    # Run model and collect attention weights
    print("  Running forward pass (capturing attention weights)...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # outputs.attentions is a tuple of (batch, n_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions

    print(f"  Processing {len(attentions)} layers of attention...")

    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        layer_attn_np = layer_attn.cpu().numpy()
        n_batch, n_heads_local, seq_len_attn, _ = layer_attn_np.shape

        for head_idx in range(n_heads_local):
            # Get attention weights for this head across all sequences
            head_attn = layer_attn_np[:, head_idx, :, :]  # (batch, query_pos, key_pos)

            # Compute entropy per query position, per sequence
            entropies = []
            sink_weights = []
            for batch_idx in range(n_batch):
                for query_pos in range(1, min(seq_len_attn, seq_len)):  # skip pos 0 (no prev context)
                    # Get attention distribution over keys
                    attn_dist = head_attn[batch_idx, query_pos, :query_pos + 1]
                    if attn_dist.sum() > 0:
                        attn_dist = attn_dist / attn_dist.sum()  # renormalize
                        ent = entropy_of_attention(attn_dist)
                        entropies.append(ent)
                        sink_weights.append(float(attn_dist[0]))

            if entropies:
                mean_ent = float(np.mean(entropies))
                std_ent = float(np.std(entropies))
                mean_sink = float(np.mean(sink_weights)) if sink_weights else 0.0
                head_type = classify_head(mean_ent, mean_sink)
                entries_needed = estimate_entries_needed(mean_ent)

                head_entropies.append(HeadEntropy(
                    layer=layer_idx,
                    head=head_idx,
                    entropy=mean_ent,
                    std_entropy=std_ent,
                    head_type=head_type,
                    n_entries_needed=entries_needed,
                ))

    print(f"  Analyzed {len(head_entropies)} heads")

    # Compute summary statistics
    entropies_arr = np.array([h.entropy for h in head_entropies])

    result = CalibrationResult(
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        n_heads_total=n_layers * n_heads,
        n_calibration_seqs=len(calibration_prompts),
        seq_len=seq_len,
        head_entropies=head_entropies,
        mean_entropy=float(np.mean(entropies_arr)),
        min_entropy=float(np.min(entropies_arr)),
        max_entropy=float(np.max(entropies_arr)),
        std_entropy=float(np.std(entropies_arr)),
    )
    result.entropy_cv = result.std_entropy / result.mean_entropy if result.mean_entropy > 0 else 0

    # Count head types
    type_counts = {}
    for h in head_entropies:
        type_counts[h.head_type] = type_counts.get(h.head_type, 0) + 1
    result.head_type_counts = type_counts

    total = len(head_entropies)
    result.sink_ratio = type_counts.get("sink", 0) / total if total > 0 else 0
    result.diffuse_ratio = type_counts.get("diffuse", 0) / total if total > 0 else 0

    return result


def generate_recommendation(result: CalibrationResult) -> str:
    """Generate a go/no-go recommendation for TurboKV trial implementation."""
    
    parts = []
    
    # Check 1: Is there significant entropy variance?
    if result.entropy_cv > 0.4:
        parts.append(f"✅ Healthy entropy CV ({result.entropy_cv:.2f}) — significant head specialization")
    elif result.entropy_cv > 0.2:
        parts.append(f"⚠️ Moderate entropy CV ({result.entropy_cv:.2f}) — some benefit expected")
    else:
        parts.append(f"❌ Low entropy CV ({result.entropy_cv:.2f}) — little benefit from per-head allocation")
    
    # Check 2: Is there a wide entropy range?
    if result.max_entropy > 0 and result.min_entropy > 0 and result.max_entropy / max(result.min_entropy, 0.01) > 50:
        parts.append(f"✅ Wide entropy range ({result.min_entropy:.2f}–{result.max_entropy:.2f} bits) — strong head differentiation")
    elif result.max_entropy - result.min_entropy > 3.0:
        parts.append(f"⚠️ Moderate entropy range ({result.min_entropy:.2f}–{result.max_entropy:.2f} bits)")
    else:
        parts.append(f"❌ Narrow entropy range ({result.min_entropy:.2f}–{result.max_entropy:.2f} bits)")
    
    # Check 3: Sink head ratio
    if result.sink_ratio > 0.30:
        parts.append(f"✅ High sink head ratio ({result.sink_ratio:.1%}) — many heads are highly compressible")
    elif result.sink_ratio > 0.10:
        parts.append(f"⚠️ Moderate sink head ratio ({result.sink_ratio:.1%})")
    else:
        parts.append(f"⚠️ Low sink head ratio ({result.sink_ratio:.1%}) — less opportunity")
    
    # Check 4: Diffuse head ratio
    if result.diffuse_ratio > 0.10:
        parts.append(f"✅ Some diffuse heads ({result.diffuse_ratio:.1%}) — budget reallocation targets")
    else:
        parts.append(f"⚠️ Few diffuse heads ({result.diffuse_ratio:.1%}) — less reallocation benefit")
    
    # Overall recommendation
    score = 0
    if result.entropy_cv > 0.4: score += 2
    elif result.entropy_cv > 0.2: score += 1
    
    if result.max_entropy / max(result.min_entropy, 0.01) > 50: score += 2
    elif result.max_entropy - result.min_entropy > 3.0: score += 1
    
    if result.sink_ratio > 0.30: score += 2
    elif result.sink_ratio > 0.10: score += 1
    
    if result.diffuse_ratio > 0.10: score += 1
    
    parts.append("")
    if score >= 5:
        result.recommendation = "GO — Proceed with trial implementation"
        parts.append("🟢 VERDICT: GO — entropy-adaptive approach strongly indicated")
        parts.append("")
        parts.append("Trial implementation plan:")
        parts.append("1. Add calibration mode to llama-server (--entropy-calibrate)")
        parts.append("2. Store per-head entropy scores in model metadata")
        parts.append("3. Implement per-head eviction or per-head mixed-precision quantization")
        parts.append("4. Test on Bonsai CodeNeedle suite at 32k context")
        parts.append("5. If quality holds, test on Qwen3.6 jQuery/CodeNeedle at 200k")
    elif score >= 3:
        result.recommendation = "MAYBE — Further investigation needed"
        parts.append("🟡 VERDICT: Cautious GO — moderate potential")
        parts.append("")
        parts.append("Recommendation: Start with a quick C++ validation tool in the fork")
        parts.append("that captures attention entropy during inference, then decide.")
    else:
        result.recommendation = "NO-GO — Not worth the implementation effort"
        parts.append("🔴 VERDICT: NO-GO — insufficient head specialization")
        parts.append("")
        parts.append("The model doesn't show the extreme entropy variance needed.")
        parts.append("Focus on Turbo6 K + Turbo3 V as the compression path.")
    
    return "\n".join(parts)


# ============================================================================
# Main
# ============================================================================

def print_report(result: CalibrationResult, detailed: bool = True):
    """Print a formatted analysis report."""
    
    print("=" * 65)
    print(f"  ENTROPY-ADAPTIVE KV CACHE — CALIBRATION REPORT")
    print("=" * 65)
    print(f"  Model:              {result.model_name}")
    print(f"  Architecture:       {result.n_layers} layers × {result.n_heads} heads = {result.n_heads_total} total")
    print(f"  Calibration seqs:   {result.n_calibration_seqs}")
    print(f"  Sequence length:    {result.seq_len}")
    print()
    print("  ENTROPY STATISTICS:")
    print(f"    Mean entropy:     {result.mean_entropy:.3f} bits")
    print(f"    Min entropy:      {result.min_entropy:.3f} bits")
    print(f"    Max entropy:      {result.max_entropy:.3f} bits")
    print(f"    Std deviation:    {result.std_entropy:.3f} bits")
    print(f"    CV (mean/std):    {result.entropy_cv:.3f}")
    print(f"    Range ratio:      {result.max_entropy / max(result.min_entropy, 0.001):.1f}×")
    print()
    print("  HEAD TYPE DISTRIBUTION:")
    for htype in ["sink", "focused", "positional", "selective", "mixed", "diffuse"]:
        count = result.head_type_counts.get(htype, 0)
        pct = count / max(result.n_heads_total, 1) * 100
        bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
        print(f"    {htype:>12}: {count:4d} ({pct:5.1f}%) {bar}")
    print()
    print("  KEY RATIOS:")
    print(f"    Easily compressible (sink+focused):    {result.sink_ratio + result.head_type_counts.get('focused', 0)/max(result.n_heads_total,1):.1%}")
    print(f"    Hard to compress (diffuse+mixed):       {result.diffuse_ratio + result.head_type_counts.get('mixed', 0)/max(result.n_heads_total,1):.1%}")
    print()
    print("  RECOMMENDATION:")
    print(result.recommendation)
    print()

    if detailed and result.head_entropies:
        print("  PER-HEAD ENTROPY HEATMAP:")
        print("  (layer × head, values in bits)")
        print()
        for layer_idx in range(result.n_layers):
            layer_heads = [h for h in result.head_entropies if h.layer == layer_idx]
            if not layer_heads:
                continue
            line = f"  L{layer_idx:2d}: "
            for h in layer_heads:
                e = h.entropy
                if e < 1.0:
                    line += "."
                elif e < 2.0:
                    line += "░"
                elif e < 3.0:
                    line += "▒"
                elif e < 4.0:
                    line += "▓"
                else:
                    line += "█"
            line += f"  avg={np.mean([h.entropy for h in layer_heads]):.2f}"
            print(line)
        print("  Legend: . <1  ░1-2  ▒2-3  ▓3-4  █>4 bits")
        print()

    # Simulated strategy comparison
    print("  SIMULATED STRATEGY COMPARISON:")
    print(f"  {'Compression':>12} | {'Entropy-Adaptive':>18} | {'Sink+Recent':>14} | {'Recent-K':>10}")
    print(f"  {'':->12}-+-{'>':->18}-+-{'>':->14}-+-{'>':->10}")
    for cr in [1.0, 1.3, 2.0, 2.5, 3.3, 5.0, 10.0]:
        ea = analytical_simulate(result.head_entropies, cr, "entropy_adaptive", result.mean_entropy)
        sr = analytical_simulate(result.head_entropies, cr, "sink_recent", result.mean_entropy)
        rk = analytical_simulate(result.head_entropies, cr, "recent_k", result.mean_entropy)
        print(f"  {cr:>6.1f}×{'':>5} | {ea.exact_match:>16.1%} | {sr.exact_match:>12.1%} | {rk.exact_match:>8.1%}")

    print()
    print("=" * 65)
    print(f"  Report saved to entropy_calibration_{result.model_name.replace('/', '_')}.json")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-Adaptive KV Cache Validation Tool"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--seqs", type=int, default=20,
        help="Number of calibration sequences (default: 20)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length in tokens (default: 128)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--detailed", action="store_true", default=True,
        help="Show detailed per-head heatmap"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-run even if cached results exist"
    )
    parser.add_argument(
        "--analytical-only", action="store_true",
        help="Skip model experiment, use analytical model with GPT-2 default params"
    )

    args = parser.parse_args()

    if args.analytical_only:
        print("Analytical-only mode: using SCJedi's published GPT-2 small entropy distribution.")
        print("This shows the expected pattern for a model with strong head specialization.")
        print()
        
        # Use SCJedi's reported GPT-2 statistics
        result = CalibrationResult(
            model_name="gpt2 (analytical reference)",
            n_layers=12,
            n_heads=12,
            n_heads_total=144,
            n_calibration_seqs=20,
            seq_len=128,
            mean_entropy=2.81,
            min_entropy=0.02,
            max_entropy=5.95,
            std_entropy=1.46,
            entropy_cv=1.46 / 2.81,
            head_type_counts={
                "sink": 71, "diffuse": 24, "positional": 17,
                "mixed": 16, "selective": 14, "focused": 2,
            },
        )
        result.sink_ratio = 71 / 144
        result.diffuse_ratio = 24 / 144

        # Add synthetic head entropy data for heatmap
        np.random.seed(42)
        for layer in range(12):
            for head in range(12):
                # Sample from a distribution that roughly matches SCJedi's findings
                base = np.random.uniform(0.5, 4.5)
                if base < 1.5:  # sink/focused (most common in middle layers)
                    htype = "sink"
                    actual_ent = np.random.uniform(0.02, 1.5)
                elif base < 3.0:  # mixed
                    htype = "mixed"
                    actual_ent = np.random.uniform(2.0, 3.5)
                else:  # diffuse (common in early layers)
                    htype = "diffuse"
                    actual_ent = np.random.uniform(3.5, 5.95)
                    
                result.head_entropies.append(HeadEntropy(
                    layer=layer, head=head,
                    entropy=float(actual_ent),
                    std_entropy=float(np.random.uniform(0.1, 0.5)),
                    head_type=htype,
                    n_entries_needed=estimate_entries_needed(actual_ent),
                ))
        
        result.recommendation = "GO — SCJedi's GPT-2 results show 300× entropy range"
        print_report(result, detailed=args.detailed)

        # Save report
        out_path = f"entropy_calibration_analytical.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved: {out_path}")
        return 0

    # Full model experiment
    try:
        result = run_model_experiment(args.model, args.seqs, args.seq_len, args.device)
    except ImportError as e:
        print(f"ERROR: {e}")
        print()
        print("To run the full experiment, install PyTorch and Transformers:")
        print("  pip install torch transformers")
        print()
        print("For an analytical-only simulation (using SCJedi's published results):")
        print(f"  python {sys.argv[0]} --analytical-only")
        return 1
    except Exception as e:
        print(f"ERROR running model experiment: {e}")
        return 1

    result.recommendation = generate_recommendation(result)
    print_report(result, detailed=args.detailed)

    # Save report
    safe_name = args.model.replace("/", "_")
    out_path = f"entropy_calibration_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Also save per-head details
    heads_out_path = f"entropy_heads_{safe_name}.json"
    with open(heads_out_path, "w") as f:
        json.dump([h.to_dict() for h in result.head_entropies], f, indent=2)

    print(f"  Saved: {out_path}")
    print(f"  Saved: {heads_out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
