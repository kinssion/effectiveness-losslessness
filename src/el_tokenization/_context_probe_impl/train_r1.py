from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import POP909RelationDataset, TARGET_SLOT_END, TARGET_SLOT_START
from .model import ContextAudiationExpert


class TargetOnlyBaseline(nn.Module):
    """Audit whether target distribution alone leaks the adjacency label."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        target = slots[:, TARGET_SLOT_START:TARGET_SLOT_END]
        summary = torch.cat(
            (target.mean(dim=1), target.std(dim=1, unbiased=False)), dim=-1
        )
        return self.network(summary).squeeze(-1)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def relation_tensors(model: ContextAudiationExpert, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    output = model(batch, log_snr=torch.full((batch.shape[0],), 20.0, device=batch.device))
    combined = 0.5 * (output.phrase_relation + output.broad_context_relation)
    return {
        "phrase": output.phrase_relation,
        "broad": output.broad_context_relation,
        "combined": combined,
    }


def pair_loss(
    positive: dict[str, torch.Tensor],
    negative: dict[str, torch.Tensor],
) -> torch.Tensor:
    return 0.5 * (
        F.softplus(-(positive["phrase"] - negative["phrase"])).mean()
        + F.softplus(-(positive["broad"] - negative["broad"])).mean()
    )


def weighted_negative_loss(
    positive: dict[str, torch.Tensor],
    same: dict[str, torch.Tensor],
    cross: dict[str, torch.Tensor],
    *,
    same_weight: float,
    cross_weight: float,
) -> torch.Tensor:
    denominator = same_weight + cross_weight
    if denominator <= 0:
        raise ValueError("at least one negative loss weight must be positive")
    return (
        same_weight * pair_loss(positive, same)
        + cross_weight * pair_loss(positive, cross)
    ) / denominator


def empty_metrics() -> dict[str, float]:
    result: dict[str, float] = {"examples": 0.0, "loss_sum": 0.0}
    for kind in ("same", "cross"):
        for head in ("phrase", "broad", "combined"):
            result[f"{kind}_{head}_correct"] = 0.0
            result[f"{kind}_{head}_margin_sum"] = 0.0
    return result


def finalize_metrics(raw: dict[str, float]) -> dict[str, float]:
    examples = max(raw["examples"], 1.0)
    output = {"examples": int(raw["examples"]), "loss": raw["loss_sum"] / examples}
    for kind in ("same", "cross"):
        for head in ("phrase", "broad", "combined"):
            output[f"{kind}_{head}_accuracy"] = (
                raw[f"{kind}_{head}_correct"] / examples
            )
            output[f"{kind}_{head}_mean_margin"] = (
                raw[f"{kind}_{head}_margin_sum"] / examples
            )
    output["mean_combined_accuracy"] = 0.5 * (
        output["same_combined_accuracy"] + output["cross_combined_accuracy"]
    )
    return output


def update_metrics(
    metrics: dict[str, float],
    positive: dict[str, torch.Tensor],
    same: dict[str, torch.Tensor],
    cross: dict[str, torch.Tensor],
    loss: torch.Tensor,
) -> None:
    count = positive["combined"].shape[0]
    metrics["examples"] += count
    metrics["loss_sum"] += float(loss.detach()) * count
    for kind, negative in (("same", same), ("cross", cross)):
        for head in ("phrase", "broad", "combined"):
            margin = positive[head].detach() - negative[head].detach()
            metrics[f"{kind}_{head}_correct"] += float((margin > 0).sum())
            metrics[f"{kind}_{head}_margin_sum"] += float(margin.sum())


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
        for key in ("positive", "same_song_mismatch", "cross_song_mismatch")
    )


def split_scores(
    scores: dict[str, torch.Tensor], batch_size: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    groups = []
    for offset in range(3):
        groups.append(
            {
                key: value[offset * batch_size : (offset + 1) * batch_size]
                for key, value in scores.items()
            }
        )
    return groups[0], groups[1], groups[2]


@torch.no_grad()
def evaluate_full(
    model: ContextAudiationExpert,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = empty_metrics()
    for batch in loader:
        positive, same, cross = move_batch(batch, device)
        batch_size = positive.shape[0]
        scores = relation_tensors(model, torch.cat((positive, same, cross), dim=0))
        pos_scores, same_scores, cross_scores = split_scores(scores, batch_size)
        loss = 0.5 * (
            pair_loss(pos_scores, same_scores) + pair_loss(pos_scores, cross_scores)
        )
        update_metrics(metrics, pos_scores, same_scores, cross_scores, loss)
    return finalize_metrics(metrics)


@torch.no_grad()
def evaluate_target_only(
    model: TargetOnlyBaseline,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = empty_metrics()
    for batch in loader:
        positive, same, cross = move_batch(batch, device)
        batch_size = positive.shape[0]
        values = model(torch.cat((positive, same, cross), dim=0))
        pos, same_value, cross_value = values.split(batch_size)
        pos_scores = {head: pos for head in ("phrase", "broad", "combined")}
        same_scores = {
            head: same_value for head in ("phrase", "broad", "combined")
        }
        cross_scores = {
            head: cross_value for head in ("phrase", "broad", "combined")
        }
        loss = 0.5 * (
            F.softplus(-(pos - same_value)).mean()
            + F.softplus(-(pos - cross_value)).mean()
        )
        update_metrics(metrics, pos_scores, same_scores, cross_scores, loss)
    return finalize_metrics(metrics)


def train_epoch_full(
    model: ContextAudiationExpert,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_gradient_norm: float,
    same_loss_weight: float,
    cross_loss_weight: float,
) -> dict[str, float]:
    model.train()
    metrics = empty_metrics()
    for batch in loader:
        positive, same, cross = move_batch(batch, device)
        batch_size = positive.shape[0]
        scores = relation_tensors(model, torch.cat((positive, same, cross), dim=0))
        pos_scores, same_scores, cross_scores = split_scores(scores, batch_size)
        loss = weighted_negative_loss(
            pos_scores,
            same_scores,
            cross_scores,
            same_weight=same_loss_weight,
            cross_weight=cross_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        update_metrics(metrics, pos_scores, same_scores, cross_scores, loss)
    return finalize_metrics(metrics)


def train_epoch_target_only(
    model: TargetOnlyBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_gradient_norm: float,
    same_loss_weight: float,
    cross_loss_weight: float,
) -> dict[str, float]:
    model.train()
    metrics = empty_metrics()
    for batch in loader:
        positive, same, cross = move_batch(batch, device)
        batch_size = positive.shape[0]
        values = model(torch.cat((positive, same, cross), dim=0))
        pos, same_value, cross_value = values.split(batch_size)
        denominator = same_loss_weight + cross_loss_weight
        loss = (
            same_loss_weight * F.softplus(-(pos - same_value)).mean()
            + cross_loss_weight * F.softplus(-(pos - cross_value)).mean()
        ) / denominator
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        pos_scores = {head: pos for head in ("phrase", "broad", "combined")}
        same_scores = {
            head: same_value for head in ("phrase", "broad", "combined")
        }
        cross_scores = {
            head: cross_value for head in ("phrase", "broad", "combined")
        }
        update_metrics(metrics, pos_scores, same_scores, cross_scores, loss)
    return finalize_metrics(metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--same-loss-weight", type=float, default=1.0)
    parser.add_argument("--cross-loss-weight", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("R1 training requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    datasets = {
        "train": POP909RelationDataset(
            args.data_root,
            split="train",
            seed=args.seed,
            max_windows=args.max_train_windows,
        ),
        "validation": POP909RelationDataset(
            args.data_root,
            split="validation",
            seed=args.seed + 1,
            max_windows=args.max_validation_windows,
        ),
        "test": POP909RelationDataset(
            args.data_root,
            split="test",
            seed=args.seed + 2,
            max_windows=args.max_test_windows,
        ),
    }
    loader_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=loader_generator if split == "train" else None,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }

    full = ContextAudiationExpert(
        model_dim=args.model_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        feedforward_dim=3 * args.model_dim,
        dropout=0.1,
    ).to(device)
    # target_prior has no justified R1 relation target; keep it inert rather
    # than accidentally training it through a future refactor.
    for parameter in full.target_prior_head.parameters():
        parameter.requires_grad_(False)
    baseline = TargetOnlyBaseline(hidden_dim=args.model_dim).to(device)
    full_optimizer = torch.optim.AdamW(
        (parameter for parameter in full.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    baseline_optimizer = torch.optim.AdamW(
        baseline.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_full = train_epoch_full(
            full,
            loaders["train"],
            full_optimizer,
            device,
            args.max_gradient_norm,
            args.same_loss_weight,
            args.cross_loss_weight,
        )
        train_baseline = train_epoch_target_only(
            baseline,
            loaders["train"],
            baseline_optimizer,
            device,
            args.max_gradient_norm,
            args.same_loss_weight,
            args.cross_loss_weight,
        )
        validation_full = evaluate_full(full, loaders["validation"], device)
        validation_baseline = evaluate_target_only(
            baseline, loaders["validation"], device
        )
        row = {
            "epoch": epoch,
            "train_full": train_full,
            "train_target_only": train_baseline,
            "validation_full": validation_full,
            "validation_target_only": validation_baseline,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        selection_score = validation_full["same_combined_accuracy"]
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": full.state_dict(),
                    "model_config": {
                        "model_dim": args.model_dim,
                        "n_heads": args.heads,
                        "n_layers": args.layers,
                        "feedforward_dim": 3 * args.model_dim,
                        "dropout": 0.1,
                    },
                    "epoch": epoch,
                    "validation": validation_full,
                },
                args.output / "best_context_expert.pt",
            )
            torch.save(
                {
                    "model_state_dict": baseline.state_dict(),
                    "epoch": epoch,
                    "validation": validation_baseline,
                },
                args.output / "target_only_baseline.pt",
            )

    checkpoint = torch.load(
        args.output / "best_context_expert.pt",
        map_location=device,
        weights_only=False,
    )
    full.load_state_dict(checkpoint["model_state_dict"])
    baseline_checkpoint = torch.load(
        args.output / "target_only_baseline.pt",
        map_location=device,
        weights_only=False,
    )
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"])
    test_full = evaluate_full(full, loaders["test"], device)
    test_baseline = evaluate_target_only(baseline, loaders["test"], device)
    delta = (
        test_full["same_combined_accuracy"]
        - test_baseline["same_combined_accuracy"]
    )
    test_examples = int(test_full["examples"])

    def wilson_lower(accuracy: float, count: int, z: float = 1.96) -> float:
        if count <= 0:
            return 0.0
        denominator = 1.0 + z * z / count
        center = accuracy + z * z / (2.0 * count)
        radius = z * math.sqrt(
            accuracy * (1.0 - accuracy) / count + z * z / (4.0 * count * count)
        )
        return (center - radius) / denominator

    same_lower = wilson_lower(test_full["same_combined_accuracy"], test_examples)
    cross_lower = wilson_lower(test_full["cross_combined_accuracy"], test_examples)
    if same_lower > 0.5 and delta >= 0.03:
        verdict = "CLEAN_SAME_SONG_RELATION_SIGNAL"
    elif cross_lower > 0.5:
        verdict = "CROSS_SONG_SHORTCUT_ONLY"
    else:
        verdict = "CLEAN_RELATION_NO_SIGNAL"
    report = {
        "verdict": verdict,
        "scope": "r1_clean_same_song_primary_random_cross_audit",
        "best_epoch": best_epoch,
        "dataset_windows": {key: len(value) for key, value in datasets.items()},
        "test_full": test_full,
        "test_target_only": test_baseline,
        "test_same_accuracy_wilson_lower_95": same_lower,
        "test_cross_accuracy_wilson_lower_95": cross_lower,
        "test_full_minus_target_only_same_accuracy": delta,
        "target_prior_trained": False,
        "negative_contract": {
            "same_song_wrong_position": {
                "training_weight": args.same_loss_weight,
                "role": "primary relation objective",
            },
            "random_cross_song": {
                "training_weight": args.cross_loss_weight,
                "role": "audit only until feature-matched hard negatives exist",
            },
        },
        "limitations": [
            "cross-song negatives are not yet feature-matched hard negatives",
            "no predicted-clean noise curriculum in this run",
            "no human labels used for training",
            "human anchor audit not yet run",
        ],
        "config": vars(args) | {"data_root": str(args.data_root), "output": str(args.output)},
    }
    write_json(args.output / "history.json", history)
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
