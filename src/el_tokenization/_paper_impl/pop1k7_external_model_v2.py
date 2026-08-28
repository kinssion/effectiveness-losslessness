from __future__ import annotations

import torch

from .external_tokenizer_training import ExternalTokenizerCausalLM, SerializedLMBatch


class FastExternalTokenizerCausalLM(ExternalTokenizerCausalLM):
    """Exact contiguous-prefix causal fast path for serialized Pop1K7 rows."""

    def hidden_sequence(self, batch: SerializedLMBatch) -> torch.Tensor:
        values = self.token_embedding(batch.input_ids)
        position = self._sequence_position(
            values.shape[1], device=values.device, dtype=values.dtype
        )
        values = self.input_projection(values + position.unsqueeze(0))
        values = values.masked_fill(~batch.valid.unsqueeze(-1), 0.0)
        length = int(values.shape[1])
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=values.device),
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
        return self.output_normalization(values)
