from __future__ import annotations


def paper_serialized_transformer_class() -> type:
    try:
        from .._paper_impl.pop1k7_external_model_v2 import FastExternalTokenizerCausalLM
    except ImportError as error:
        raise RuntimeError(
            "install the 'paper' extra to use checkpoint-compatible models"
        ) from error
    return FastExternalTokenizerCausalLM
