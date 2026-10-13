"""A tiny randomly initialised causal LM + byte-level BPE tokenizer for offline tests of the HF code paths."""

from __future__ import annotations

from typing import Any, Sequence


def tiny_tokenizer(corpus: Sequence[str], vocab_size: int = 800) -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|pad|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(list(corpus), trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>")


def tiny_model(vocab_size: int, seed: int = 0) -> Any:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg)
