import torch
import torch.nn.functional as F

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50) -> int:
    """
    Sample a token from the given logits (shape: [vocab_size]) using
    temperature, top-k, or top-p (nucleus) filtering.
    """
    if temperature == 0.0:
        return torch.argmax(logits).item()
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k
    if top_k > 0:
        val, _ = torch.topk(logits, top_k)
        logits[logits < val[-1]] = -float("Inf")
        
    # Apply top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p (nucleus)
        # We shift the mask to keep the first token exceeding the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
