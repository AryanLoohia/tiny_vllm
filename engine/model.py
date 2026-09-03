import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMModel:
    """A thin wrapper around a HuggingFace model + tokenizer.

    Handles device selection (MPS on Apple Silicon, CPU otherwise) and the
    precision modes:
      - "fp32": full precision (default)
      - "fp16": half precision (smaller, faster on MPS)
      - "int8": dynamic 8-bit quantization of Linear layers
    """

    def __init__(self, model_id="HuggingFaceTB/SmolLM2-135M-Instruct", precision="fp32"):
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
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"[MODEL] int8 quantization unavailable, using fp32: {e}")
                self.precision = "fp32"

        self.model.eval()

    def reload(self, precision=None, model_id=None):
        """Reload the model, optionally with a new precision and/or model id."""
        if precision is not None:
            self.precision = precision
        if model_id is not None:
            self.model_id = model_id
        del self.model
        self._load_model()
        print(f"[MODEL] Reloaded model_id={self.model_id} precision={self.precision}")

    def tokenize(self, text, format_chat=True):
        """Turn a string into a flat list of token ids."""
        tokens = None
        if format_chat:
            try:
                tokens = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            except Exception:
                tokens = None

        if tokens is None:
            tokens = self.tokenizer.encode(text)

        # Normalise whatever HF gave us into a flat list of ints.
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

    def decode_token(self, token_id):
        return self.tokenizer.decode([token_id])

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

    def kv_bytes_per_token(self):
        """Bytes of KV cache memory used per generated token, per request."""
        try:
            cfg = self.model.config
            num_layers = cfg.num_hidden_layers
            num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            dtype_bytes = 2 if self.precision == "fp16" else 4
            return num_layers * num_kv_heads * head_dim * 2 * dtype_bytes
        except Exception:
            return 0
