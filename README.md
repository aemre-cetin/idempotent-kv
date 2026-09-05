# idempotent-kv: Zero-Copy In-Place KV-Cache Compaction for LLMs

[![Patent Pending](https://img.shields.io/badge/Patent-Pending%20(US%2064%2F148%2C668)-blue.svg)](https://uspto.gov)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Tested%20on-NVIDIA%20Blackwell%20sm__120-purple.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)]()

> **Eliminate 100% of auxiliary VRAM allocations during KV-cache context eviction in vLLM, SGLang, and Hugging Face inference engines.**

---

## 🚀 The Bottleneck: The Memory Wall

In long-context autoregressive LLM inference (32k to 128k+ tokens), intermediate Key and Value activations consume tens of gigabytes of accelerator memory. To prevent out-of-memory (OOM) faults, dynamic eviction policies discard low-importance context tokens.

However, standard deep learning runtimes execute compaction using out-of-place gathering (`torch.gather` / `cudaMalloc`):
1. **Auxiliary VRAM Spikes:** Allocating secondary buffers ($O(N \cdot D)$) triggers transient OOM faults.
2. **Memory Bus Congestion:** Double-buffering saturates HBM bandwidth with redundant read/write transactions.
3. **OS Allocation Latency Jitter:** Dynamic `cudaMalloc` calls introduce unpredictable stalls in real-time serving.

---

## ⚡ The Solution: In-Situ Idempotent Permutations

`idempotent-kv` reorganizes multi-dimensional tensors **directly within existing memory allocations** using **$O(1)$ scalar hardware registers**:
- **Idempotent Invariant:** Enforces $f(f(x)) = f(x)$, locking retained tokens into stabilized attractor basins $[0, K-1]$.
- **In-Register 2-Cycle Fast-Path:** Swaps mutually transposed tokens directly within thread registers.
- **Bitmask-Free Cycle Follower:** Identifies cycle leaders on-the-fly via strictly minimal index traversal.
- **100% Zero Auxiliary VRAM:** Exactly **0.00 MB** secondary memory allocated.
- **Bit-Exact Numerical Parity:** Zero approximation error ($\Delta = 0.000000$, 0 NaN).

---

## 📊 Benchmark: NVIDIA RTX PRO 500 Blackwell (`sm_120`)

*Workload: Batch=1, Heads=32, SeqLen=8,192, HeadDim=128, Capacity=4,096 (50% Eviction, float16)*

| Implementation | Latency (ms) | Peak Aux VRAM | VRAM Saved | Numerical Diff |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch Out-of-Place** | 3.137 ms | 96.00 MB | Baseline | 0.000000 |
| **`idempotent-kv` (Ours)** | **2.016 ms** | **0.00 MB** | **96.00 MB (100%)** | **0.000000** |
| **Improvement** | **1.56x Faster** | **0.00 MB** | **100% Eliminated** | **Bit-Exact** |

---

## 📦 Installation

```bash
git clone https://github.com/aemre-cetin/idempotent-kv.git
cd idempotent-kv
pip install -e .
```

Requirements: `torch >= 2.0.0`, `triton >= 2.1.0`.

---

## 🛠️ Quickstart

```python
import torch
from idempotent_kv import InplaceKVCompactor

compactor = InplaceKVCompactor()

# Key and Value caches [Batch, Heads, SeqLen, HeadDim] on GPU
key_cache = torch.randn((1, 32, 8192, 128), dtype=torch.float16, device="cuda")
value_cache = torch.randn((1, 32, 8192, 128), dtype=torch.float16, device="cuda")

# Top-k active token indices to retain
active_indices = torch.randperm(8192, device="cuda")[:4096]

# Build idempotent permutation map f(x)
target_map = compactor.build_idempotent_map(
    batch=1, heads=32, seq_len=8192, 
    active_indices=active_indices, capacity=4096, device="cuda"
)

# In-place compaction: zero secondary global memory allocated
compacted_k, compacted_v = compactor.compact(
    key_cache, value_cache, target_map, capacity=4096
)
# Returns key_cache[:, :, :4096, :] and value_cache[:, :, :4096, :]
```

### vLLM Integration Hook

```python
from idempotent_kv.integrations import VLLMInplaceCompactionHook

# Attach directly to vLLM attention worker
compaction_hook = VLLMInplaceCompactionHook(capacity=4096, protected_prefix_len=4)

# In-place context eviction during forward pass
compacted_k, compacted_v = compaction_hook(
    key_cache, value_cache, attention_scores=cumulative_attention_weights
)
```

---

## 🛡️ Patent & Intellectual Property Notice

The mathematical formulations, state-transition architectures, and in-situ hardware compaction kernels implemented in this library are protected under pending patent application with the United States Patent and Trademark Office:

* **U.S. Patent Application Number:** **`64/148,668`**
* **Confirmation Number:** **`5890`**
* **Status:** **PATENT PENDING**
* **First Named Inventor:** **Dr. Ahmet Emre ÇETİN**

Academic evaluation, non-commercial research, and open-source collaboration are permitted under the terms of the Apache 2.0 License. Commercial deployment in proprietary hardware or commercial cloud runtimes is subject to licensing agreements with the author.

---

## 📜 Academic Citation

```bibtex
@article{cetin2026idempotentkv,
  title={Zero-Copy In-Place Compaction and Idempotent Associative Routing of Dynamic Key-Value Cache Tensors in Deep Learning Accelerators},
  author={Cetin, A. Emre},
  journal={arXiv preprint},
  year={2026},
  note={U.S. Patent Application No. 64/148,668}
}

@article{cetin2013idempotent,
  title={Idempotent Permutations},
  author={Cetin, A. E.},
  journal={arXiv:1307.3877 [cs.DS]},
  year={2013}
}
```

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright © 2026 Dr. A. Emre ÇETİN. All Rights Reserved.

