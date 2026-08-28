from __future__ import annotations


def paper_event_transformer_class() -> type:
    try:
        from .._paper_impl.pop1k7_model_v2 import FastCausalUnifiedMatchedMusicModel
    except ImportError as error:
        raise RuntimeError(
            "install the 'paper' extra to use checkpoint-compatible models"
        ) from error
    return FastCausalUnifiedMatchedMusicModel
