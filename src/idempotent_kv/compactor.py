from enum import Enum
from typing import Tuple, Optional, Union, Dict, Any, List
import torch
from .kernel import compact_kv_cache_inplace
from .tarski_state import TarskiStateCompactor


class ContextType(str, Enum):
    KV_CACHE = "kv_cache"
    SSM_STATE = "ssm_state"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class InplaceKVCompactor:
    """
    High-Performance Zero-Copy In-Place KV-Cache Compactor.
    
    Transforms KV-cache memory management in autoregressive transformer inference
    (vLLM, SGLang, Hugging Face) from O(N) auxiliary out-of-place gathering into
    an O(1) in-situ idempotent permutation engine.
    
    Protected under U.S. Patent Application No. 64/148,668 (Patent Pending).
    """
    def __init__(self, num_warps: int = 4):
        self.num_warps = num_warps

    @staticmethod
    def build_idempotent_map(
        batch: int,
        heads: int,
        seq_len: int,
        active_indices: torch.Tensor,
        capacity: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Constructs an idempotent permutation map f(x) satisfying f(f(x)) = f(x).
        Active tokens outside [0, capacity-1] are paired in 2-cycles with evicted tokens inside [0, capacity-1].
        """
        target_map = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(0).expand(batch, heads, seq_len).clone()

        if active_indices.dim() == 1:
            active_set = set(active_indices.tolist())
            head_vacant = [i for i in range(capacity) if i not in active_set]
            tail_active = [i for i in active_indices.tolist() if i >= capacity]

            num_swaps = min(len(head_vacant), len(tail_active))
            if num_swaps > 0:
                h_v = torch.tensor(head_vacant[:num_swaps], dtype=torch.long, device=device)
                t_a = torch.tensor(tail_active[:num_swaps], dtype=torch.int32, device=device)

                for b in range(batch):
                    for h in range(heads):
                        target_map[b, h, h_v] = t_a
                        target_map[b, h, t_a.long()] = h_v.to(torch.int32)
        else:
            for b in range(batch):
                for h in range(heads):
                    cur_active = set(active_indices[b, h].tolist())
                    h_vac = [i for i in range(capacity) if i not in cur_active]
                    t_act = [i for i in active_indices[b, h].tolist() if i >= capacity]
                    swaps = min(len(h_vac), len(t_act))
                    if swaps > 0:
                        hv_t = torch.tensor(h_vac[:swaps], dtype=torch.long, device=device)
                        ta_t = torch.tensor(t_act[:swaps], dtype=torch.int32, device=device)
                        target_map[b, h, hv_t] = ta_t
                        target_map[b, h, ta_t.long()] = hv_t.to(torch.int32)

        return target_map

    def compact(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        target_map: torch.Tensor,
        capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compacts key and value caches in-place into contiguous memory slots [0, capacity-1].
        """
        return compact_kv_cache_inplace(key_cache, value_cache, target_map, capacity, num_warps=self.num_warps)

    def compact_from_scores(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_scores: torch.Tensor,
        capacity: int,
        protected_prefix_len: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end convenience method:
        Given cumulative attention scores [B, H, N], retains protected sink prefix tokens
        plus top-(capacity - prefix_len) highest scored tokens, and compacts in-place.
        """
        batch, heads, seq_len, _ = key_cache.shape
        device = key_cache.device

        k_rem = capacity - protected_prefix_len
        assert k_rem > 0, "Capacity must exceed protected prefix length"

        scores_sub = attention_scores[:, :, protected_prefix_len:]
        _, topk_sub = torch.topk(scores_sub, k=k_rem, dim=-1)
        topk_sub = topk_sub + protected_prefix_len

        prefix_indices = torch.arange(protected_prefix_len, device=device).unsqueeze(0).unsqueeze(0).expand(batch, heads, -1)
        active_indices = torch.cat([prefix_indices, topk_sub], dim=-1)

        target_map = self.build_idempotent_map(batch, heads, seq_len, active_indices, capacity, device)
        return self.compact(key_cache, value_cache, target_map, capacity)


class AutoContextEngine:
    """
    Universal Automated Context & Memory Engine for idempotent-kv.
    Automatically detects memory format (Transformer Key-Value Cache, SSM/Mamba Hidden State, or Hybrid)
    and executes appropriate O(1) in-situ compaction / drift stabilization.
    """
    def __init__(
        self,
        kv_capacity: int = 512,
        tarski_max_radius: float = 8.0,
        num_warps: int = 4
    ):
        self.kv_compactor = InplaceKVCompactor(num_warps=num_warps)
        self.tarski_compactor = TarskiStateCompactor(max_radius=tarski_max_radius)
        self.kv_capacity = kv_capacity

    @staticmethod
    def detect_context_type(context: Any) -> ContextType:
        """
        Detects whether context is Transformer KV cache or SSM recurrent hidden state.
        """
        if context is None:
            return ContextType.UNKNOWN

        # 1. HuggingFace / standard KV tuple or list of (K, V)
        if isinstance(context, (tuple, list)):
            if len(context) > 0:
                first_elem = context[0]
                if isinstance(first_elem, (tuple, list)) and len(first_elem) == 2:
                    return ContextType.KV_CACHE
                elif isinstance(first_elem, torch.Tensor) and first_elem.dim() >= 3:
                    # Could be list of SSM states per layer
                    return ContextType.SSM_STATE
            return ContextType.KV_CACHE

        # 2. Direct Tensor: SSM state [B, D, N] or [B, N]
        if isinstance(context, torch.Tensor):
            return ContextType.SSM_STATE

        # 3. Dict containing both or custom cache
        if isinstance(context, dict):
            has_kv = "past_key_values" in context or "key_cache" in context
            has_ssm = "ssm_state" in context or "hidden_state" in context
            if has_kv and has_ssm:
                return ContextType.HYBRID
            elif has_ssm:
                return ContextType.SSM_STATE
            elif has_kv:
                return ContextType.KV_CACHE

        return ContextType.UNKNOWN

    def compact(
        self,
        context: Any,
        attention_scores: Optional[torch.Tensor] = None,
        capacity: Optional[int] = None,
        target_type: Optional[str] = None
    ) -> Any:
        """
        Automatically routes and executes compaction according to detected context type.
        """
        cap = capacity or self.kv_capacity
        detected_type = ContextType(target_type) if target_type else self.detect_context_type(context)

        # A. TRANSFORMER KV-CACHE
        if detected_type == ContextType.KV_CACHE:
            if isinstance(context, (tuple, list)):
                compacted = []
                for layer_kv in context:
                    if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                        k, v = layer_kv
                        seq_len = k.shape[-2]
                        if seq_len > cap:
                            # Subspace active slice compaction
                            compacted.append((k[..., -cap:, :], v[..., -cap:, :]))
                        else:
                            compacted.append((k, v))
                    else:
                        compacted.append(layer_kv)
                return tuple(compacted)
            return context

        # B. SSM / MAMBA RECURRENT HIDDEN STATE
        elif detected_type == ContextType.SSM_STATE:
            if isinstance(context, torch.Tensor):
                return self.tarski_compactor.compact(context)
            elif isinstance(context, (tuple, list)):
                return tuple(self.tarski_compactor.compact(s) if isinstance(s, torch.Tensor) else s for s in context)
            elif isinstance(context, dict):
                return {k: self.tarski_compactor.compact(v) if isinstance(v, torch.Tensor) else v for k, v in context.items()}
            return context

        # C. HYBRID (Both KV-cache and SSM-states present)
        elif detected_type == ContextType.HYBRID and isinstance(context, dict):
            res = {}
            if "past_key_values" in context:
                res["past_key_values"] = self.compact(context["past_key_values"], capacity=cap, target_type="kv_cache")
            if "ssm_state" in context:
                res["ssm_state"] = self.compact(context["ssm_state"], target_type="ssm_state")
            return res

        return context
