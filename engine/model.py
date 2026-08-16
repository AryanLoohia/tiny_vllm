import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMModel:
    """
    Wraps the Hugging Face model and tokenizer, abstracting hardware device
    selection (MPS/CPU) and basic tokenization operations.

    Precision modes (`precision`):
      - "fp32":  Default full-precision weights.
      - "fp16":  Half-precision weights (smaller memory, faster on MPS).
      - "int8":  Dynamic 8-bit quantization of nn.Linear layers (CPU fallback
                 friendly; demonstrates a quantization comparison).
    """
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
                 precision: str = "fp32"):
        self.model_id = model_id
        self.precision = precision
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._load_model()

    def _load_model(self):
        kwargs = {}
        if self.precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.precision == "fp16":
            try:
                self.model = self.model.to(self.device, dtype=torch.float16)
            except Exception:
                self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        if self.precision == "int8":
            # Dynamic quantization of Linear layers (works on CPU; falls back gracefully)
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"[MODEL] int8 quantization unavailable, using fp32: {e}")
                self.precision = "fp32"
        self.model.eval()

    def reload(self, precision: str = None, model_id: str = None):
        """Reload the model, optionally with a new precision and/or model id."""
        if precision is not None:
            self.precision = precision
        if model_id is not None:
            self.model_id = model_id
        del self.model
        self._load_model()
        print(f"[MODEL] Reloaded model_id={self.model_id} precision={self.precision}")

    def tokenize(self, text: str, format_chat: bool = True) -> list[int]:
        tokens = None
        if format_chat:
            try:
                tokens = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True
                )
            except Exception:
                pass
        if tokens is None:
            tokens = self.tokenizer.encode(text)

        if isinstance(tokens, dict) or hasattr(tokens, "keys"):
            if "input_ids" in tokens:
                tokens = tokens["input_ids"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], str):
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens)
        return tokens

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def get_eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache memory consumed per generated token, per request."""
        try:
            cfg = self.model.config
            num_layers = cfg.num_hidden_layers
            num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            dtype_bytes = 2 if self.precision == "fp16" else 4
            return num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
        except Exception:
            return 0
