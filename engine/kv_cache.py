import torch
from engine.request import Request


def pad_and_batch_kv_cache(requests, device):
    """Left-pad every request's KV cache to the same length and stack them.

    Returns:
        batched_pkv   : tuple of (key, value) per layer, each shaped
                        [batch, num_kv_heads, max_cache_len, head_dim]
        position_ids  : [batch, 1] next position for each request
        attention_mask: [batch, max_cache_len + 1] (1 = real token, 0 = padding)
    """
    for r in requests:
        assert r.kv_cache is not None, \
            f"Request {r.request_id} must be prefilled before batching"

    num_layers = len(requests[0].kv_cache)
    cache_lengths = [r.kv_cache[0][0].shape[2] for r in requests]
    max_cache_len = max(cache_lengths)

    padded_pkv = []
    for layer_idx in range(num_layers):
        keys = []
        values = []
        for r in requests:
            k, v = r.kv_cache[layer_idx]
            seq_len_i = k.shape[2]
            pad_len = max_cache_len - seq_len_i

            if pad_len > 0:
                pad_shape = (1, k.shape[1], pad_len, k.shape[3])
                pad_k = torch.zeros(pad_shape, device=device, dtype=k.dtype)
                pad_v = torch.zeros(pad_shape, device=device, dtype=v.dtype)
                k = torch.cat([pad_k, k], dim=2)
                v = torch.cat([pad_v, v], dim=2)

            keys.append(k)
            values.append(v)

        batch_k = torch.cat(keys, dim=0)
        batch_v = torch.cat(values, dim=0)
        padded_pkv.append((batch_k, batch_v))

    batched_pkv = tuple(padded_pkv)

    # Next position for each request is its current length (0-indexed -> seq_len-1).
    position_ids = torch.tensor(
        [[r.seq_len - 1] for r in requests], dtype=torch.long, device=device
    )

    # Attention mask covers the cache plus the one new token we're about to feed.
    masks = []
    for r in requests:
        seq_len_i = r.kv_cache[0][0].shape[2]
        pad_len = max_cache_len - seq_len_i
        left = torch.zeros(pad_len, device=device)
        right = torch.ones(seq_len_i + 1, device=device)
        masks.append(torch.cat([left, right]))
    attention_mask = torch.stack(masks).long()

    return batched_pkv, position_ids, attention_mask


def _extract_kv_from_output(batch_pkv):
    """Handle both DynamicCache and plain tuples, return a list of (k, v)."""
    if hasattr(batch_pkv, "key_cache") and hasattr(batch_pkv, "value_cache"):
        return [
            (batch_pkv.key_cache[i], batch_pkv.value_cache[i])
            for i in range(len(batch_pkv.key_cache))
        ]
    return [(layer[0], layer[1]) for layer in batch_pkv]


def slice_and_update_kv_cache(batch_pkv, requests):
    """Strip the padding off the batched cache and write each slice back.

    After the forward pass the cache has grown by one token, so the real part
    for request i is the slice [pad_len : pad_len + seq_len_i + 1].
    """
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
            k_i = k_batch[r_idx:r_idx + 1, :, pad_len:, :]
            v_i = v_batch[r_idx:r_idx + 1, :, pad_len:, :]
            new_kv.append((k_i, v_i))

        r.kv_cache = new_kv
