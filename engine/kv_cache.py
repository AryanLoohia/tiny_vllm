import torch
from typing import List, Tuple
from engine.request import Request


def pad_and_batch_kv_cache(
    requests: List[Request], 
    device: torch.device
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor, torch.Tensor]:
    """
    Batches the KV cache of multiple requests together by left-padding them
    to the maximum cache sequence length.
    
    Returns:
        batched_pkv: Tuple of (key_layer, value_layer) for each layer.
            Each tensor has shape [batch_size, num_kv_heads, max_cache_len, head_dim].
        position_ids: Tensor of shape [batch_size, 1] containing the next position index for each request.
        attention_mask: Tensor of shape [batch_size, max_cache_len + 1] containing 1s for valid tokens and 0s for padding.
    """
    for r in requests:
        assert r.kv_cache is not None, f"Request {r.request_id} KV cache must be prefilled before decoding steps"
        
    num_layers = len(requests[0].kv_cache)
    cache_lengths = [r.kv_cache[0][0].shape[2] for r in requests]
    max_cache_len = max(cache_lengths)
    
    padded_pkv = []
    for layer_idx in range(num_layers):
        keys = []
        values = []
        for r_idx, r in enumerate(requests):
            k, v = r.kv_cache[layer_idx]  # Shape: [1, num_kv_heads, seq_len_i, head_dim]
            seq_len_i = k.shape[2]
            pad_len = max_cache_len - seq_len_i
            
            if pad_len > 0:
                # Left-pad with zeros
                pad_shape = (1, k.shape[1], pad_len, k.shape[3])
                pad_k = torch.zeros(pad_shape, device=device, dtype=k.dtype)
                pad_v = torch.zeros(pad_shape, device=device, dtype=v.dtype)
                k_padded = torch.cat([pad_k, k], dim=2)
                v_padded = torch.cat([pad_v, v], dim=2)
            else:
                k_padded, v_padded = k, v
                
            keys.append(k_padded)
            values.append(v_padded)
            
        batch_k = torch.cat(keys, dim=0)
        batch_v = torch.cat(values, dim=0)
        padded_pkv.append((batch_k, batch_v))
        
    batched_pkv = tuple(padded_pkv)
    
    # position_ids shape: [batch_size, 1]. The position of the next token for request i is its current seq_len - 1.
    position_ids = torch.tensor([[r.seq_len - 1] for r in requests], dtype=torch.long, device=device)
    
    # attention_mask shape: [batch_size, max_cache_len + 1]
    masks = []
    for r in requests:
        seq_len_i = r.kv_cache[0][0].shape[2]
        pad_len = max_cache_len - seq_len_i
        mask_i = torch.cat([
            torch.zeros(pad_len, device=device),
            torch.ones(seq_len_i + 1, device=device)
        ])
        masks.append(mask_i)
    attention_mask = torch.stack(masks).long()
    
    return batched_pkv, position_ids, attention_mask


def _extract_kv_from_output(batch_pkv) -> list:
    """
    Extract KV cache from model output, handling both DynamicCache and plain tuples.
    Returns list of (key, value) tuples per layer.
    """
    if hasattr(batch_pkv, 'key_cache') and hasattr(batch_pkv, 'value_cache'):
        return [
            (batch_pkv.key_cache[i], batch_pkv.value_cache[i])
            for i in range(len(batch_pkv.key_cache))
        ]
    return [(layer[0], layer[1]) for layer in batch_pkv]


def slice_and_update_kv_cache(
    batch_pkv,
    requests: List[Request]
) -> None:
    """
    Slices the updated batched KV cache (which now has length max_cache_len + 1)
    and saves the unpadded caches back to each request.
    
    Handles both DynamicCache and tuple-of-tuples formats.
    """
    # Normalize to list of (key, value) tuples
    layers = _extract_kv_from_output(batch_pkv)
    num_layers = len(layers)
    cache_lengths = [r.kv_cache[0][0].shape[2] for r in requests]
    max_cache_len = max(cache_lengths)
    
    for r_idx, r in enumerate(requests):
        seq_len_i = cache_lengths[r_idx]
        pad_len = max_cache_len - seq_len_i
        
        new_kv = []
        for layer_idx in range(num_layers):
            k_batch, v_batch = layers[layer_idx]
            
            # Slice request index out of batch, and discard the left padding
            k_i = k_batch[r_idx:r_idx+1, :, pad_len:, :]
            v_i = v_batch[r_idx:r_idx+1, :, pad_len:, :]
            new_kv.append((k_i, v_i))
            
        r.kv_cache = new_kv
