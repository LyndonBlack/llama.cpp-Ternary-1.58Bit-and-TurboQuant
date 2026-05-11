#!/usr/bin/env python3
"""
Extract K projection weights from a Qwen GGUF model and generate realistic
synthetic K values to test pre-RoPE vs post-RoPE turbo4 quantization.
"""
import sys, struct, random, math
from pathlib import Path
import numpy as np

def read_gguf_tensor(f, offset, n_bytes):
    """Read raw bytes from a GGUF file at the given offset."""
    f.seek(offset)
    return f.read(n_bytes)

def find_tensor_in_gguf(gguf_path, tensor_name):
    """Find a tensor in a GGUF file and return its (offset, n_elements, dtype)."""
    with open(gguf_path, 'rb') as f:
        # Read magic
        magic = f.read(4)
        if magic == b'GGUF':
            version = struct.unpack('<I', f.read(4))[0]
        else:
            raise ValueError(f"Not a GGUF file (magic: {magic})")
        
        # Read header
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_metadata = struct.unpack('<Q', f.read(8))[0]
        
        # Skip metadata
        for _ in range(n_metadata):
            # Read key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            # Read value type
            val_type = struct.unpack('<I', f.read(4))[0]
            # Skip value based on type
            if val_type == 8:  # string
                s_len = struct.unpack('<Q', f.read(8))[0]
                f.read(s_len)
            elif val_type == 0:  # uint8
                f.read(1)
            elif val_type in (1, 2, 3, 4, 5):  # int8, int16, int32, float16, float32
                sizes = {1:1, 2:2, 3:4, 4:2, 5:4}
                f.read(sizes[val_type])
            elif val_type == 6:  # bool
                f.read(1)
            elif val_type == 7:  # array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 8:  # string array
                    for _ in range(arr_len):
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(s_len)
            elif val_type == 9:  # int64
                f.read(8)
            elif val_type == 10:  # float64
                f.read(8)
            elif val_type == 11:  # uint32
                f.read(4)
            elif val_type == 12:  # uint64
                f.read(8)
            else:
                raise ValueError(f"Unknown metadata type {val_type}")
        
        # Read tensor info
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{"Q"*n_dims}', f.read(8*n_dims))
            ggml_type = struct.unpack('<I', f.read(4))[0]
            tensor_offset = struct.unpack('<Q', f.read(8))[0]
            
            if name == tensor_name:
                return (tensor_offset, dims, ggml_type)
    
    return None

# GGML type to numpy dtype mapping
GGML_TO_NP = {
    0: np.float32,  # GGML_TYPE_F32
    1: np.float16,  # GGML_TYPE_F16
    7: np.float16,  # GGML_TYPE_Q8_0 reads as bytes, special handling
}

def parse_ggml_type(dtype_id):
    """Parse GGML type enum."""
    names = {
        0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
        8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
        13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
        18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
        23: 'IQ4_XS', 24: 'F16', 25: 'BF16', 26: 'TQ1_0', 27: 'TQ2_0',
    }
    return names.get(dtype_id, f'UNKNOWN({dtype_id})')

def dequantize_tensor(data, ggml_type, dims):
    """Dequantize a tensor from GGML format to float32 numpy array."""
    if ggml_type == 0:  # F32
        return np.frombuffer(data, dtype=np.float32).reshape(dims)
    elif ggml_type == 1:  # F16
        return np.frombuffer(data, dtype=np.float16).reshape(dims).astype(np.float32)
    elif ggml_type == 24:  # F16 (same as 1)
        return np.frombuffer(data, dtype=np.float16).reshape(dims).astype(np.float32)
    else:
        return None  # Cannot easily dequantize

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/lyndon/AI/models/Qwen3.6-35B-A3B-Q5_K_M.gguf'
    
    print(f"Reading model: {model_path}")
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
    
    # Known tensor names for Qwen3 architecture
    # Layer 0 K projection: blk.0.attn_k.weight (or similar)
    search_names = [
        "blk.0.attn_k.weight",
        "blk.0.attn_qkv.weight",  # fused QKV
        "model.layers.0.self_attn.k_proj.weight",
    ]
    
    found = None
    for name in search_names:
        result = find_tensor_in_gguf(model_path, name)
        if result:
            found = (name, result)
            print(f"Found tensor: {name}")
            break
    
    if not found:
        # Search for any attention weight in layer 0
        print("Searching for attention weights in layer 0...")
        # Try common patterns
        with open(model_path, 'rb') as f:
            data = f.read()
        
        # Quick prefix scan for blk.0
        prefixes = [b'blk.0.', b'model.layers.0.']
        idx = 0
        found_name = None
        while idx < len(data) - 100:
            for prefix in prefixes:
                if data[idx:idx+len(prefix)] == prefix:
                    # Found a tensor name starting with this prefix
                    end = data.find(b'\x00', idx)
                    if end > idx:
                        name = data[idx:end].decode('utf-8', errors='replace')
                        if 'attn' in name.lower() and ('k' in name.lower() or 'key' in name.lower()):
                            found_name = name
                            break
            if found_name:
                break
            idx += 1
        
        if found_name:
            result = find_tensor_in_gguf(model_path, found_name)
            found = (found_name, result)
            print(f"Found: {found_name}")
        else:
            print("Could not find K projection tensor in GGUF file.")
            print("\nFalling back to statistical simulation with per-dimension variance.")
            return simulate_k_values()

    name, (offset, dims, ggml_type) = found
    print(f"  offset: {offset}, shape: {dims}, type: {GGML_TO_NP.get(ggml_type, parse_ggml_type(ggml_type))}")
    
    # Read and dequantize
    n_bytes = 1
    for d in dims:
        n_bytes *= d
    if ggml_type in (0, 1, 24):  # F32 or F16
        if ggml_type == 0:
            n_elements = n_bytes // 4
        else:
            n_elements = n_bytes // 2
    else:
        print(f"Cannot dequantize quantized weights. Using weight statistics only.")
        return simulate_k_values_from_stats(dims)
    
    # Read raw data
    with open(model_path, 'rb') as f:
        f.seek(offset)
        raw = f.read(n_bytes)
    
    weights = dequantize_tensor(raw, ggml_type, dims)
    if weights is None:
        print("Could not dequantize. Using fallback.")
        return simulate_k_values()
    
    print(f"Weight shape: {weights.shape}")
    
    # For fused QKV: extract K portion
    # QKV is usually [n_embd, n_embd*3] — K is the second third
    if len(weights.shape) == 2 and 'qkv' in name.lower():
        n_cols = weights.shape[1]
        k_start = n_cols // 3
        k_end = 2 * n_cols // 3
        k_weights = weights[:, k_start:k_end]
        print(f"Extracted K weights from fused QKV: {k_weights.shape}")
    else:
        k_weights = weights
    
    # Generate realistic K values: random_input @ K_weights + bias
    # With n_embd=4096 for Qwen3.6-8B or n_embd=2560 for smaller
    n_embd = k_weights.shape[0]
    n_head_kv_dim = k_weights.shape[1]
    print(f"n_embd={n_embd}, n_head_kv_dim={n_head_kv_dim}")
    
    # Generate synthetic tokens through K projection
    np.random.seed(42)
    n_tokens = 64
    n_dim = min(128, n_head_kv_dim)  # head dim
    
    # Generate synthetic hidden states (what a transformer layer outputs)
    hidden = np.random.randn(n_tokens, n_embd).astype(np.float32)
    
    # Compute K = hidden @ k_weights
    K_raw = hidden @ k_weights  # (n_tokens, n_head_kv_dim)
    
    # We need per-head K values (each head is n_rot dims)
    # Usually n_head * n_rot = n_head_kv_dim
    # For Qwen3.6 with GQA=4 and n_rot=128: n_head_kv = 8, n_head_kv_dim = 1024
    # But we're looking at a single layer's K output which is n_head_kv * n_rot
    
    # Take first n_dim values (simulating single head or first head)
    K_head = K_raw[:, :n_dim]
    
    print(f"Generated K values: {K_head.shape}")
    print(f"K stats: mean={K_head.mean():.4f}, std={K_head.std():.4f}")
    print(f"Per-dim std: min={K_head.std(axis=0).min():.4f}, max={K_head.std(axis=0).max():.4f}")
    
    # Save to binary file for the C++ validation tool
    outpath = "/tmp/qwen_kvalues.bin"
    K_head.astype(np.float32).tofile(outpath)
    print(f"\nSaved {n_tokens} x {n_dim} K values to {outpath}")
    print(f"   File size: {n_tokens * n_dim * 4} bytes")
    
    # Also compute and print SNR using our own numpy implementation
    print("\nRunning pre-RoPE vs post-RoPE comparison...")
    freq_base = 1000000.0
    freq_scale = 0.25
    n_rot = n_dim
    
    for t in range(n_tokens):
        k = K_head[t].copy()
        
        # Pre-RoPE: quantize (simulate with noise), dequant, RoPE
        # Turbo4 at 4 bits = 16 centroids. 
        # The quantization error is roughly uniform [-scale/32, scale/32] per block
        # Simulate: quantize and dequantize using the actual block structure
        
        # Actually, let's use optimal quantization: scale to [-1, 1], round to 16 levels
        block_size = 128  # QK_TURBO4
        k_pre_quant = k.copy()
        k_post_quant = k.copy()
        
        # Per-block quantization (turbo4 sim)
        for b in range(0, n_rot, block_size):
            be = min(b + block_size, n_rot)
            blk = k[b:be]
            
            # Simulate turbo4: find norm, assign to 16 centroids, reconstruct
            norm = np.max(np.abs(blk))
            if norm > 1e-10:
                # Quantize: map to 4-bit indices [-8, 7]
                idx = np.clip(np.round(blk * 8 / norm), -7, 7).astype(np.int8)
                # Dequantize
                k_pre_quant[b:be] = idx.astype(np.float32) * norm / 8
        
        # Pre-RoPE: apply RoPE to quantized values
        k_pre_rope = k_pre_quant.copy()
        for i in range(0, n_rot, 2):
            theta = t * freq_base ** (-2.0 * i / n_rot) * freq_scale
            c, s = math.cos(theta), math.sin(theta)
            v0, v1 = k_pre_rope[i], k_pre_rope[i+1]
            k_pre_rope[i] = v0 * c - v1 * s
            k_pre_rope[i+1] = v0 * s + v1 * c
        
        # Post-RoPE: apply RoPE first
        k_rope = k.copy()
        for i in range(0, n_rot, 2):
            theta = t * freq_base ** (-2.0 * i / n_rot) * freq_scale
            c, s = math.cos(theta), math.sin(theta)
            v0, v1 = k_rope[i], k_rope[i+1]
            k_rope[i] = v0 * c - v1 * s
            k_rope[i+1] = v0 * s + v1 * c
        
        # Quantize post-RoPE
        k_post_rope = k_rope.copy()
        for b in range(0, n_rot, block_size):
            be = min(b + block_size, n_rot)
            blk = k_rope[b:be]
            norm = np.max(np.abs(blk))
            if norm > 1e-10:
                idx = np.clip(np.round(blk * 8 / norm), -7, 7).astype(np.int8)
                k_post_rope[b:be] = idx.astype(np.float32) * norm / 8
        
        # Reference: RoPE without quantization
        k_ref = k.copy()
        for i in range(0, n_rot, 2):
            theta = t * freq_base ** (-2.0 * i / n_rot) * freq_scale
            c, s = math.cos(theta), math.sin(theta)
            v0, v1 = k_ref[i], k_ref[i+1]
            k_ref[i] = v0 * c - v1 * s
            k_ref[i+1] = v0 * s + v1 * c
        
        # SNR
        signal_power = np.sum(k_ref ** 2)
        noise_pre = np.sum((k_ref - k_pre_rope) ** 2)
        noise_post = np.sum((k_ref - k_post_rope) ** 2)
        snr_pre = 10 * np.log10(signal_power / noise_pre) if noise_pre > 0 else 200
        snr_post = 10 * np.log10(signal_power / noise_post) if noise_post > 0 else 200
        
        if t < 5 or t % 10 == 0:
            print(f"  t={t:3d}  pre: {snr_pre:7.2f} dB  post: {snr_post:7.2f} dB  Δ: {snr_pre - snr_post:+.2f} dB")
    
    print("\nDone. K values saved for C++ validation tool.")


def simulate_k_values():
    """Fallback: generate realistic K values."""
    n_tokens = 64
    n_dim = 128
    np.random.seed(42)
    
    # Realistic: per-dimension variance decays exponentially
    dim_var = np.exp(-np.arange(n_dim) / 16)  # ~5% at dim 48
    K = np.random.randn(n_tokens, n_dim) * np.sqrt(dim_var)
    
    outpath = "/tmp/qwen_kvalues.bin"
    K.astype(np.float32).tofile(outpath)
    print(f"Simulated realistic K values saved to {outpath}")
    print(f"  n_tokens={n_tokens}, n_dim={n_dim}")


def simulate_k_values_from_stats(dims):
    """Fallback with model weight dimensions."""
    print(f"Model weight dims suggest n_embd={dims[0]}, n_head_kv_dim={dims[1]}")
    simulate_k_values()


if __name__ == '__main__':
    main()
