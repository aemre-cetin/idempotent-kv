import torch
from typing import Optional
from ..compactor import InplaceKVCompactor

class VLLMInplaceCompactionHook:
    """
    Drop-in compaction hook for vLLM Attention and CacheEngine workers.
    
    Intercepts dynamic context eviction phases and replaces secondary
    buffer allocations with O(1) in-situ idempotent swaps.
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
        active_indices: Optional[torch.Tensor] = None
    ):
        batch, heads, seq_len, _ = key_cache.shape
        device = key_cache.device

        if seq_len <= self.capacity:
            return key_cache, value_cache

        if active_indices is not None:
            target_map = self.compactor.build_idempotent_map(
                batch, heads, seq_len, active_indices, self.capacity, device
            )
            return self.compactor.compact(key_cache, value_cache, target_map, self.capacity)
        elif attention_scores is not None:
            return self.compactor.compact_from_scores(
                key_cache, value_cache, attention_scores, self.capacity, self.protected_prefix_len
            )
        else:
            raise ValueError("Either attention_scores or active_indices must be provided for compaction.")
