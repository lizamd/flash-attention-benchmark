# Flash Attention Kernel Benchmark Report
## AMD MI300X (gfx942) - CK vs Triton vs Native PyTorch

### Executive Summary

This report documents the benchmarking and tuning of flash attention kernels on AMD MI300X GPU using AITER (AMD Inference Triton Extension for ROCm). We compared three implementations:

1. **CK (Composable Kernel)** - Hand-optimized C++/HIP kernels
2. **Triton** - Python DSL compiled to GPU kernels
3. **Native PyTorch** - Standard O(N²) attention implementation

**Key Finding**: After tuning, Triton outperforms CK at small sequence lengths (seq=512), while CK dominates at longer sequences.

---

### Environment Setup

#### Hardware
- **GPU**: AMD Instinct MI300X
- **Architecture**: gfx942
- **LDS (Shared Memory)**: 64KB per workgroup

#### Software
- **AITER**: AMD's flash attention library
- **PyTorch**: 2.x with ROCm support
- **Triton**: AMD ROCm Triton

#### Installation Steps

```bash
# Clone AITER repository
git clone https://github.com/ROCm/aiter.git
cd aiter

# Install for MI300X (gfx942)
GPU_ARCHS="gfx942" python3 setup.py install
```

---

### Benchmark Configuration

#### Test Parameters
```python
configs = [
    (batch=4, seq=512,  heads=8, d_k=256),
    (batch=4, seq=1024, heads=8, d_k=256),
    (batch=4, seq=2048, heads=8, d_k=256),
    (batch=4, seq=4096, heads=8, d_k=256),
]
dtype = torch.float16
causal = True
warmup = 10 iterations
benchmark = 100 iterations
```

#### Benchmark Code

```python
"""
Benchmark: CK vs Triton vs PyTorch Native Attention
Save as: bench_attention.py
"""

import torch
import torch.nn.functional as F
import time
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def native_attention(q, k, v, causal=True):
    """Standard PyTorch attention (O(N^2) memory)"""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if causal:
        seq_len = q.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def benchmark(fn, q, k, v, warmup=10, iters=100):
    """Benchmark a function"""
    for _ in range(warmup):
        out = fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        out = fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms per iteration


# Import backends
from aiter import flash_attn_func as ck_flash_attn
from aiter.ops.triton.mha import flash_attn_func as triton_flash_attn

def ck_attention(q, k, v):
    out, _ = ck_flash_attn(q, k, v, causal=True, return_lse=True)
    return out

def triton_attention(q, k, v):
    out, _ = triton_flash_attn(q, k, v, causal=True, return_lse=True)
    return out


# Run benchmarks
configs = [
    (4, 512, 8, 256),
    (4, 1024, 8, 256),
    (4, 2048, 8, 256),
    (4, 4096, 8, 256),
]

for batch, seq, heads, d_k in configs:
    # Flash attention format: (batch, seq, heads, d_k) in fp16
    q = torch.randn(batch, seq, heads, d_k, device=device, dtype=torch.float16)
    k = torch.randn(batch, seq, heads, d_k, device=device, dtype=torch.float16)
    v = torch.randn(batch, seq, heads, d_k, device=device, dtype=torch.float16)

    # Native needs (batch, heads, seq, d_k) in fp32
    q_native = q.transpose(1, 2).float()
    k_native = k.transpose(1, 2).float()
    v_native = v.transpose(1, 2).float()

    native_ms = benchmark(native_attention, q_native, k_native, v_native)
    ck_ms = benchmark(ck_attention, q, k, v)
    triton_ms = benchmark(triton_attention, q, k, v)

    print(f"Config ({batch}, {seq}, {heads}, {d_k}):")
    print(f"  Native:  {native_ms:.3f} ms")
    print(f"  CK:      {ck_ms:.3f} ms  ({native_ms/ck_ms:.2f}x speedup)")
    print(f"  Triton:  {triton_ms:.3f} ms  ({native_ms/triton_ms:.2f}x speedup)")
```

---

### Results

#### Before Tuning (Default Triton Config)

Default config: `BLOCK_M=128, BLOCK_N=64, PRELOAD_V=false`

| Config (B,S,H,D) | Native (ms) | CK (ms) | Triton (ms) | CK Speedup | Triton Speedup |
|------------------|-------------|---------|-------------|------------|----------------|
| (4, 512, 8, 256) | 0.274 | 0.183 | 0.436 | 1.50x | 0.63x |
| (4, 1024, 8, 256) | 0.897 | 0.232 | 0.546 | 3.86x | 1.64x |
| (4, 2048, 8, 256) | 3.077 | 0.450 | 1.040 | 6.83x | 2.96x |
| (4, 4096, 8, 256) | 10.303 | 1.002 | 2.672 | 10.28x | 3.86x |

**Observation**: Triton was significantly slower than CK, especially at d_k=256.

#### After Tuning (Optimized Triton Config)

Tuned config: `BLOCK_M=64, BLOCK_N=32, PRELOAD_V=true`

| Config (B,S,H,D) | Native (ms) | CK (ms) | Triton (ms) | CK Speedup | Triton Speedup |
|------------------|-------------|---------|-------------|------------|----------------|
| (4, 512, 8, 256) | 0.278 | 0.181 | **0.162** | 1.54x | **1.71x** |
| (4, 1024, 8, 256) | 0.900 | **0.228** | 0.235 | 3.94x | 3.83x |
| (4, 2048, 8, 256) | 3.073 | **0.447** | 0.511 | 6.87x | 6.01x |
| (4, 4096, 8, 256) | 10.267 | **1.008** | 1.368 | 10.19x | 7.51x |

**Key Improvements**:
- Triton at seq=512: **0.436ms → 0.162ms** (2.7x faster)
- Triton now **beats CK** at seq=512 (0.162ms vs 0.181ms)

---

### Tuning Details

#### Config File Location
```
aiter/aiter/ops/triton/configs/gfx942-MHA-DEFAULT.json
```

#### Original Config (Default)
```json
{
  "fwd": {
    "default": {
      "BLOCK_M": 128,
      "BLOCK_N": 64,
      "PRELOAD_V": false,
      "waves_per_eu": 2,
      "num_warps": 4,
      "num_ctas": 1,
      "num_stages": 1
    }
  }
}
```

#### Tuned Config
```json
{
  "fwd": {
    "default": {
      "BLOCK_M": 64,
      "BLOCK_N": 32,
      "PRELOAD_V": true,
      "waves_per_eu": 2,
      "num_warps": 4,
      "num_ctas": 1,
      "num_stages": 1
    }
  }
}
```

#### Why This Works

| Parameter | Original | Tuned | Rationale |
|-----------|----------|-------|-----------|
| BLOCK_M | 128 | 64 | Smaller tiles reduce LDS pressure at d_k=256 |
| BLOCK_N | 64 | 32 | Reduces shared memory from ~49KB to ~24KB |
| PRELOAD_V | false | true | Prefetch V tensor to hide memory latency |

**Shared Memory Calculation**:
```
Original: BLOCK_M × d_k + BLOCK_N × d_k = 128×256 + 64×256 = 49,152 bytes
Tuned:    BLOCK_M × d_k + BLOCK_N × d_k = 64×256 + 32×256 = 24,576 bytes
```

The MI300X has 64KB LDS limit. Smaller tiles leave more room for other data and allow better occupancy.

---

### Reproduce Steps

#### Step 1: Modify Triton Config
```bash
# Edit the config file
vim /path/to/aiter/aiter/ops/triton/configs/gfx942-MHA-DEFAULT.json

# Change "default" section:
# BLOCK_M: 128 -> 64
# BLOCK_N: 64 -> 32
# PRELOAD_V: false -> true
```

#### Step 2: Reinstall AITER
```bash
cd /path/to/aiter
GPU_ARCHS="gfx942" python3 setup.py install
```

#### Step 3: Run Benchmark
```bash
python3 bench_attention.py
```

#### Alternative: Enable Triton Autotune
```bash
# Instead of manual tuning, use built-in autotune
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1
python3 bench_attention.py
```

---

### Analysis

#### CK vs Triton Performance Characteristics

| Sequence Length | Winner | Reason |
|-----------------|--------|--------|
| 512 | **Triton** | Smaller tiles have less overhead at short sequences |
| 1024+ | **CK** | Hand-tuned memory access patterns, adaptive tile sizes |

#### Head Dimension Impact

- **d_k=64/128** (typical): CK ≈ Triton (both well-optimized)
- **d_k=256** (large): CK > Triton (CK adapts tile sizes per head_dim)
- **d_k>256**: Both fail (exceeds 64KB LDS limit)

#### Recommendations

1. **For production with typical configs (d_k=64-128)**:
   - Use either CK or Triton - performance is similar
   - CK is default in AITER

2. **For large head dimensions (d_k=256)**:
   - Use CK for seq > 512
   - Use tuned Triton for seq ≤ 512

3. **Dynamic backend selection**:
```python
def choose_backend(seq_len, head_dim):
    if seq_len <= 512 and head_dim >= 256:
        return triton_flash_attn  # Tuned for small seq / large head
    else:
        return ck_flash_attn      # Better for longer sequences
```

---

### Additional Notes

#### Flash Attention Key Concepts

1. **Tiling**: Process Q, K, V in blocks to fit in LDS
2. **Online Softmax**: Compute softmax incrementally without storing full attention matrix
3. **LSE (Log-Sum-Exp)**: Store `log(sum(exp(scores)))` per row for backward pass
4. **Causal Masking**: Skip upper-triangular computations for decoder attention

#### Why Flash Attention is Faster

| Aspect | Native Attention | Flash Attention |
|--------|------------------|-----------------|
| Memory | O(N²) - stores full attention matrix | O(N) - tiled computation |
| Memory BW | High - multiple HBM round trips | Low - stays in LDS |
| Scaling | OOM at seq > 4K | Works at seq > 128K |

#### Hardware Constraints (MI300X)

- LDS per workgroup: 64KB
- Max head_dim for flash attention: ~256 (limited by LDS)
- Typical configs: d_k=64 (GPT), d_k=128 (LLaMA)

---

### References

- [AITER GitHub](https://github.com/ROCm/aiter)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Composable Kernel](https://github.com/ROCm/composable_kernel)
- [Triton Language](https://triton-lang.org/)
- [Helion DSL](https://helionlang.com/) - For writing new fused kernels

---

### Appendix: Full Benchmark Output

```
Device: cuda
GPU: AMD Instinct MI300X

==========================================================================================
Config               Native (ms)     CK (ms)         Triton (ms)     CK Speedup   Triton Speedup
==========================================================================================
(4, 512, 8, 256)     0.278           0.181           0.162           1.54x        1.71x
(4, 1024, 8, 256)    0.900           0.228           0.235           3.94x        3.83x
(4, 2048, 8, 256)    3.073           0.447           0.511           6.87x        6.01x
(4, 4096, 8, 256)    10.267          1.008           1.368           10.19x       7.51x
==========================================================================================

CK vs Triton comparison (seq=1024, d_k=64):
  CK:     0.154 ms
  Triton: 0.152 ms
  CK is 0.99x slower than Triton
```

---

*Report generated: 2025-12-30*
*Author: Benchmark and tuning performed with Claude Code assistance*
