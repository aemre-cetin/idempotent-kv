import torch
import triton
import triton.language as tl

@triton.jit
def _inplace_kv_compact_opt_kernel(
    K_ptr,                 # [B, H, N, D]
    V_ptr,                 # [B, H, N, D]
    TargetMap_ptr,         # [B, H, N]
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh, stride_mn,
    N: tl.constexpr,       # e.g. 8192, 32768
    HEAD_DIM: tl.constexpr # e.g. 128
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)

    k_base_ptr = K_ptr + pid_batch * stride_kb + pid_head * stride_kh
    v_base_ptr = V_ptr + pid_batch * stride_vb + pid_head * stride_vh
    map_base_ptr = TargetMap_ptr + pid_batch * stride_mb + pid_head * stride_mh

    offs_d = tl.arange(0, HEAD_DIM)

    # In-place cyclic permutation with strictly O(1) auxiliary registers
    for i in range(0, N):
        dest_idx = tl.load(map_base_ptr + i * stride_mn)

        if dest_idx != i:
            # 2-cycle transposition fast-path
            is_two_cycle = False
            if dest_idx > i:
                second_hop = tl.load(map_base_ptr + dest_idx * stride_mn)
                if second_hop == i:
                    is_two_cycle = True

            if is_two_cycle:
                src_k_ptr = k_base_ptr + i * stride_kn + offs_d * stride_kd
                src_v_ptr = v_base_ptr + i * stride_vn + offs_d * stride_vd
                dst_k_ptr = k_base_ptr + dest_idx * stride_kn + offs_d * stride_kd
                dst_v_ptr = v_base_ptr + dest_idx * stride_vn + offs_d * stride_vd

                v_k_i = tl.load(src_k_ptr)
                v_v_i = tl.load(src_v_ptr)
                v_k_d = tl.load(dst_k_ptr)
                v_v_d = tl.load(dst_v_ptr)

                tl.store(src_k_ptr, v_k_d)
                tl.store(src_v_ptr, v_v_d)
                tl.store(dst_k_ptr, v_k_i)
                tl.store(dst_v_ptr, v_v_i)
            else:
                # General disjoint cycle leader search (strictly minimal index traversal)
                if dest_idx > i:
                    curr = dest_idx
                    is_leader = True
                    keep_searching = True

                    while keep_searching:
                        if curr == i:
                            keep_searching = False
                        else:
                            if curr < i:
                                is_leader = False
                                keep_searching = False
                            else:
                                curr = tl.load(map_base_ptr + curr * stride_mn)

                    if is_leader:
                        src_k_ptr = k_base_ptr + i * stride_kn + offs_d * stride_kd
                        src_v_ptr = v_base_ptr + i * stride_vn + offs_d * stride_vd

                        temp_k = tl.load(src_k_ptr)
                        temp_v = tl.load(src_v_ptr)

                        curr_slot = i
                        cycle_active = True

                        while cycle_active:
                            next_slot = tl.load(map_base_ptr + curr_slot * stride_mn)

                            if next_slot == i:
                                target_k_ptr = k_base_ptr + curr_slot * stride_kn + offs_d * stride_kd
                                target_v_ptr = v_base_ptr + curr_slot * stride_vn + offs_d * stride_vd
                                tl.store(target_k_ptr, temp_k)
                                tl.store(target_v_ptr, temp_v)
                                cycle_active = False
                            else:
                                from_k_ptr = k_base_ptr + next_slot * stride_kn + offs_d * stride_kd
                                from_v_ptr = v_base_ptr + next_slot * stride_vn + offs_d * stride_vd

                                to_k_ptr = k_base_ptr + curr_slot * stride_kn + offs_d * stride_kd
                                to_v_ptr = v_base_ptr + curr_slot * stride_vn + offs_d * stride_vd

                                val_k = tl.load(from_k_ptr)
                                val_v = tl.load(from_v_ptr)

                                tl.store(to_k_ptr, val_k)
                                tl.store(to_v_ptr, val_v)

                                curr_slot = next_slot


def compact_kv_cache_inplace(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    target_map: torch.Tensor,
    compacted_capacity: int,
    num_warps: int = 4
):
    """
    Executes in-place zero-copy KV cache compaction on GPU tensors.
    
    Args:
        key_cache: [B, H, N, D] float16 or bfloat16 contiguous tensor
        value_cache: [B, H, N, D] float16 or bfloat16 contiguous tensor
        target_map: [B, H, N] int32 permutation map
        compacted_capacity: number of retained context slots
        num_warps: Triton execution warps per block
    
    Returns:
        (compacted_key, compacted_value) sliced directly as [B, H, :capacity, D]
    """
    assert key_cache.is_contiguous() and value_cache.is_contiguous(), "Key and Value caches must be contiguous"
    assert target_map.is_contiguous(), "Target map must be contiguous"
    
    batch_size, num_heads, seq_len, head_dim = key_cache.shape
    grid = (batch_size, num_heads)

    _inplace_kv_compact_opt_kernel[grid](
        key_cache, value_cache, target_map,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        target_map.stride(0), target_map.stride(1), target_map.stride(2),
        N=seq_len,
        HEAD_DIM=head_dim,
        num_warps=num_warps
    )
    return key_cache[:, :, :compacted_capacity, :], value_cache[:, :, :compacted_capacity, :]
