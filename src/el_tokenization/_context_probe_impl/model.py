from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ContextAudiationOutput:
    """Separated evidence; callers must not silently collapse it to taste."""

    phrase_relation: torch.Tensor
    broad_context_relation: torch.Tensor
    target_prior: torch.Tensor
    region_embeddings: dict[str, torch.Tensor]
    region_attention: dict[str, torch.Tensor]


class _RelationHead(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        # Keep directional left/right evidence rather than averaging it away.
        feature_dim = 8 * model_dim
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2 * model_dim),
            nn.GELU(),
            nn.Linear(2 * model_dim, 1),
        )

    def forward(
        self,
        left: torch.Tensor,
        target: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                left,
                target,
                right,
                target * left,
                target * right,
                torch.abs(target - left),
                torch.abs(target - right),
                right - left,
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class ContextAudiationExpert(nn.Module):
    """Small multiscale relation network over normalized PhraseLDM slots.

    The fixed semantic layout is eight bars of left context, four target bars,
    and four bars of right context. Raw PhraseVAE outputs must first be divided
    by the active fixslot checkpoint's ``scale_factor`` so clean training and
    sampler predicted-clean states share one coordinate system. Each bar has
    the PhraseLDM fixslot layout:
    three content slots followed by one bar-boundary slot.

    The class is intentionally only an architecture primitive.  It contains no
    claim that untrained outputs are musical scores and no baked-in weighting
    that would collapse relation, intrinsic prior, and novelty.
    """

    n_bars = 16
    slots_per_bar = 4
    latent_dim = 64
    n_slots = n_bars * slots_per_bar

    def __init__(
        self,
        *,
        model_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % n_heads:
            raise ValueError("model_dim must be divisible by n_heads")
        self.model_dim = model_dim
        self.input_projection = nn.Linear(self.latent_dim, model_dim)
        self.segment_embedding = nn.Embedding(3, model_dim)
        self.bar_embedding = nn.Embedding(self.n_bars, model_dim)
        self.slot_embedding = nn.Embedding(self.slots_per_bar, model_dim)
        self.log_snr_embedding = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(model_dim)

        self.region_names = ("left_near", "left_broad", "target", "right")
        self.region_queries = nn.Parameter(
            torch.empty(len(self.region_names), model_dim)
        )
        nn.init.normal_(self.region_queries, std=model_dim**-0.5)

        region_masks = torch.zeros(len(self.region_names), self.n_slots, dtype=torch.bool)
        region_masks[0, 4 * self.slots_per_bar : 8 * self.slots_per_bar] = True
        region_masks[1, 0 : 8 * self.slots_per_bar] = True
        region_masks[2, 8 * self.slots_per_bar : 12 * self.slots_per_bar] = True
        region_masks[3, 12 * self.slots_per_bar : 16 * self.slots_per_bar] = True
        self.register_buffer("region_masks", region_masks, persistent=False)

        segment_ids = torch.empty(self.n_slots, dtype=torch.long)
        segment_ids[: 8 * self.slots_per_bar] = 0
        segment_ids[8 * self.slots_per_bar : 12 * self.slots_per_bar] = 1
        segment_ids[12 * self.slots_per_bar :] = 2
        self.register_buffer("segment_ids", segment_ids, persistent=False)
        self.register_buffer(
            "bar_ids",
            torch.arange(self.n_bars).repeat_interleave(self.slots_per_bar),
            persistent=False,
        )
        self.register_buffer(
            "slot_ids",
            torch.arange(self.slots_per_bar).repeat(self.n_bars),
            persistent=False,
        )

        self.phrase_relation_head = _RelationHead(model_dim)
        self.broad_relation_head = _RelationHead(model_dim)
        self.target_prior_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    @classmethod
    def target_slot_mask(cls, *, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(cls.n_slots, dtype=torch.bool, device=device)
        mask[8 * cls.slots_per_bar : 12 * cls.slots_per_bar] = True
        return mask

    def _validate(
        self,
        slot_latents: torch.Tensor,
        log_snr: torch.Tensor | None,
    ) -> torch.Tensor:
        if slot_latents.ndim == 4:
            expected = (self.n_bars, self.slots_per_bar, self.latent_dim)
            if tuple(slot_latents.shape[1:]) != expected:
                raise ValueError(
                    f"rank-4 input must be [batch,{expected[0]},{expected[1]},"
                    f"{expected[2]}], got {tuple(slot_latents.shape)}"
                )
            slot_latents = slot_latents.reshape(
                slot_latents.shape[0], self.n_slots, self.latent_dim
            )
        elif slot_latents.ndim == 3:
            expected = (self.n_slots, self.latent_dim)
            if tuple(slot_latents.shape[1:]) != expected:
                raise ValueError(
                    f"rank-3 input must be [batch,{expected[0]},{expected[1]}], "
                    f"got {tuple(slot_latents.shape)}"
                )
        else:
            raise ValueError("slot_latents must have rank 3 or 4")
        if not torch.is_floating_point(slot_latents):
            raise ValueError("slot_latents must be floating point")
        if not torch.isfinite(slot_latents).all():
            raise ValueError("slot_latents contains NaN or Inf")
        if log_snr is not None:
            if log_snr.ndim not in (0, 1):
                raise ValueError("log_snr must be scalar or [batch]")
            if log_snr.ndim == 1 and log_snr.shape[0] != slot_latents.shape[0]:
                raise ValueError("log_snr batch does not match slot_latents")
        return slot_latents

    def _pool_regions(
        self, encoded: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        embeddings: dict[str, torch.Tensor] = {}
        attention: dict[str, torch.Tensor] = {}
        for index, name in enumerate(self.region_names):
            query = self.region_queries[index]
            logits = torch.einsum("bsd,d->bs", encoded, query) / math.sqrt(
                self.model_dim
            )
            region_mask = self.region_masks[index].unsqueeze(0)
            logits = logits.masked_fill(~region_mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1)
            embeddings[name] = torch.einsum("bs,bsd->bd", weights, encoded)
            attention[name] = weights
        return embeddings, attention

    def forward(
        self,
        slot_latents: torch.Tensor,
        *,
        log_snr: torch.Tensor | None = None,
    ) -> ContextAudiationOutput:
        slots = self._validate(slot_latents, log_snr)
        batch = slots.shape[0]
        if log_snr is None:
            log_snr = torch.zeros(batch, device=slots.device, dtype=slots.dtype)
        elif log_snr.ndim == 0:
            log_snr = log_snr.expand(batch)
        log_snr = log_snr.to(device=slots.device, dtype=slots.dtype).clamp(-20, 20)

        hidden = self.input_projection(slots)
        hidden = hidden + self.segment_embedding(self.segment_ids)
        hidden = hidden + self.bar_embedding(self.bar_ids)
        hidden = hidden + self.slot_embedding(self.slot_ids)
        hidden = hidden + self.log_snr_embedding(log_snr[:, None]).unsqueeze(1)
        encoded = self.output_norm(self.encoder(hidden))
        regions, attention = self._pool_regions(encoded)

        phrase_relation = self.phrase_relation_head(
            regions["left_near"], regions["target"], regions["right"]
        )
        broad_context_relation = self.broad_relation_head(
            regions["left_broad"], regions["target"], regions["right"]
        )
        target_prior = self.target_prior_head(regions["target"]).squeeze(-1)
        return ContextAudiationOutput(
            phrase_relation=phrase_relation,
            broad_context_relation=broad_context_relation,
            target_prior=target_prior,
            region_embeddings=regions,
            region_attention=attention,
        )
