from __future__ import annotations

import torch

from .iclr_matched_harness import RepresentationOutput, UnifiedMatchedMusicModel
from .note_centric_music import NoteCausalBatch


class FastCausalUnifiedMatchedMusicModel(UnifiedMatchedMusicModel):
    """Mathematically equivalent causal fast path for non-relational arms.

    Every sample is right padded and every valid query occurs before its padded
    suffix.  Therefore, for a valid causal query, all reachable keys are valid;
    a per-sample padding mask is redundant.  Arms without pairwise attention
    bias can consequently use PyTorch's 2-D causal-mask fast path instead of a
    dense ``batch * heads * length * length`` floating-point mask.

    Relational arms retain the original implementation unchanged.
    """

    def hidden_sequence(
        self, batch: NoteCausalBatch
    ) -> tuple[torch.Tensor, RepresentationOutput]:
        if self.representation_config.pairwise_time_bias:
            return super().hidden_sequence(batch)

        representation = self.representation(batch)
        values = self.input_fusion(
            torch.cat(
                (
                    representation.content_features,
                    representation.unary_time_features,
                    representation.optional_reference_features,
                ),
                dim=-1,
            )
        )
        length = int(values.shape[1])
        causal_mask = torch.triu(
            torch.ones(
                (length, length), dtype=torch.bool, device=values.device
            ),
            diagonal=1,
        )
        for block in self.blocks:
            normalized = block.norm_attention(values)
            attended, _ = block.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=causal_mask,
                need_weights=False,
                is_causal=True,
            )
            values = values + attended
            values = values + block.feedforward(block.norm_feedforward(values))
            values = values.masked_fill(~batch.valid.unsqueeze(-1), 0.0)
        return self.output_normalization(values), representation
