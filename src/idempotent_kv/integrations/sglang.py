"""
SGLang Integration Layer for Zero-Copy In-Place KV-Cache Compaction.

Provides a drop-in hook for SGLang SRT (SGLang Runtime) and RadixAttention,
eliminating secondary buffer reallocations (torch.gather / cudaMalloc) during
prefix tree node compaction and long-context token eviction.

Protected under U.S. Patent Application No. 64/148,668 and 64/148,679.
Author: Dr. A. Emre ÇETİN <aemre.cetin@gmail.com>
"""

import torch
from typing import Optional, Tuple
from ..compactor import InplaceKVCompactor


class SGLangInplaceCompactionHook:
    """
    Drop-in compaction hook for SGLang SRT (SGLang Runtime) and RadixAttention.

    Intercepts RadixCache node eviction or sequence truncation phases and replaces
    out-of-place buffer reallocation with O(1) in-situ idempotent cycle transpositions.

    Key Advantages for SGLang:
    1. Zero auxiliary VRAM allocation (prevents sudden OOM spikes during heavy concurrency).
    2. Bit-exact numerical fidelity (preserves physical memory contiguity for FlashAttention / FlashInfer).
    3. Hardware accelerated on NVIDIA Blackwell sm_120 and Hopper H100.
    """

    def __init__(self, capacity: int, protected_prefix_len: int = 4):
        self.capacity = capacity
        self.protected_prefix_len = protected_prefix_len
        self.compactor = InplaceKVCompactor()

    def __call__(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        active_indices: Optional[torch.Tensor] = None,
        target_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compacts key and value caches in-place into contiguous memory slots [0, capacity-1].

        Args:
            key_cache: Key tensor of shape [B, H, N, D]
            value_cache: Value tensor of shape [B, H, N, D]
            attention_scores: Optional cumulative attention weights [B, H, N]
            active_indices: Optional tensor of token indices to retain
            target_map: Optional precomputed idempotent permutation map [B, H, N]

        Returns:
            Tuple of (compacted_key, compacted_value) sliced directly as [B, H, :capacity, D]
        """
        batch, heads, seq_len, _ = key_cache.shape
        device = key_cache.device

        if seq_len <= self.capacity:
            return key_cache, value_cache

        if target_map is not None:
            return self.compactor.compact(key_cache, value_cache, target_map, self.capacity)

        if active_indices is not None:
            t_map = self.compactor.build_idempotent_map(
                batch, heads, seq_len, active_indices, self.capacity, device
            )
            return self.compactor.compact(key_cache, value_cache, t_map, self.capacity)

        if attention_scores is not None:
            return self.compactor.compact_from_scores(
                key_cache, value_cache, attention_scores, self.capacity, self.protected_prefix_len
            )

        raise ValueError("Either target_map, active_indices, or attention_scores must be provided for compaction.")
