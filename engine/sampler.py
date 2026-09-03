import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_p=1.0, top_k=50):
    """Sample one token id from logits using temperature, top-k and top-p.

    Greedy when temperature is 0.
    """
    if temperature == 0.0:
        return torch.argmax(logits).item()

    logits = logits / temperature

    # Top-k: keep only the k highest-probability tokens.
    if top_k > 0:
        val, _ = torch.topk(logits, top_k)
        logits[logits < val[-1]] = -float("Inf")

    # Top-p (nucleus): keep the smallest set whose cumulative prob exceeds top_p.
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Mark tokens above the threshold for removal, but always keep the first
        # token that crosses it.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
