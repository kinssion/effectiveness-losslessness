from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import POP909RelationDataset
from .model import ContextAudiationExpert


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def different_song_permutation(song_ids: list[str]) -> torch.Tensor | None:
    count = len(song_ids)
    for shift in range(max(1, count // 3), count):
        candidate = [(index + shift) % count for index in range(count)]
        if all(song_ids[index] != song_ids[other] for index, other in enumerate(candidate)):
            return torch.tensor(candidate, dtype=torch.long)
    # A mixed batch may not admit one global cyclic shift. Find row-wise donors.
    candidate = []
    for index, song_id in enumerate(song_ids):
        donor = next(
            (other for other, other_id in enumerate(song_ids) if other_id != song_id),
            None,
        )
        if donor is None:
            return None
        candidate.append(donor)
    return torch.tensor(candidate, dtype=torch.long)


def replace_context(
    source: torch.Tensor,
    donor: torch.Tensor,
    *,
    replace_left: bool,
    replace_right: bool,
) -> torch.Tensor:
    result = source.clone()
    if replace_left:
        result[:, :32] = donor[:, :32]
    if replace_right:
        result[:, 48:] = donor[:, 48:]
    return result


def scores(model: ContextAudiationExpert, value: torch.Tensor) -> dict[str, torch.Tensor]:
    output = model(
        value,
        log_snr=torch.full((value.shape[0],), 20.0, device=value.device),
    )
    return {
        "phrase": output.phrase_relation,
        "broad": output.broad_context_relation,
        "combined": 0.5 * (
            output.phrase_relation + output.broad_context_relation
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("context-dependence audit requires CUDA")
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ContextAudiationExpert(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = POP909RelationDataset(
        args.data_root, split=args.split, seed=args.seed, cache_songs=64
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    modes = {
        "normal": (False, False),
        "left_shuffled": (True, False),
        "right_shuffled": (False, True),
        "both_shuffled": (True, True),
    }
    correct = {
        mode: {head: 0 for head in ("phrase", "broad", "combined")}
        for mode in modes
    }
    margins = {
        mode: {head: 0.0 for head in ("phrase", "broad", "combined")}
        for mode in modes
    }
    examples = 0
    skipped_single_song_batches = 0
    with torch.inference_mode():
        for batch in loader:
            positive = batch["positive"].to(device, dtype=torch.float32)
            negative = batch["same_song_mismatch"].to(device, dtype=torch.float32)
            permutation = different_song_permutation(list(batch["song_id"]))
            if permutation is None:
                skipped_single_song_batches += 1
                continue
            donor = positive[permutation.to(device)]
            examples += positive.shape[0]
            for mode, (shuffle_left, shuffle_right) in modes.items():
                pos_input = replace_context(
                    positive,
                    donor,
                    replace_left=shuffle_left,
                    replace_right=shuffle_right,
                )
                neg_input = replace_context(
                    negative,
                    donor,
                    replace_left=shuffle_left,
                    replace_right=shuffle_right,
                )
                merged = torch.cat((pos_input, neg_input), dim=0)
                output = scores(model, merged)
                for head, values in output.items():
                    pos_score, neg_score = values.split(positive.shape[0])
                    margin = pos_score - neg_score
                    correct[mode][head] += int((margin > 0).sum())
                    margins[mode][head] += float(margin.sum())
    if not examples:
        raise RuntimeError("no mixed-song batch available for context audit")
    metrics: dict[str, Any] = {}
    for mode in modes:
        metrics[mode] = {
            f"{head}_accuracy": correct[mode][head] / examples
            for head in ("phrase", "broad", "combined")
        } | {
            f"{head}_mean_margin": margins[mode][head] / examples
            for head in ("phrase", "broad", "combined")
        }

    normal = metrics["normal"]["combined_accuracy"]
    both = metrics["both_shuffled"]["combined_accuracy"]

    def wilson_lower(accuracy: float, count: int, z: float = 1.96) -> float:
        denominator = 1 + z * z / count
        center = accuracy + z * z / (2 * count)
        radius = z * math.sqrt(
            accuracy * (1 - accuracy) / count + z * z / (4 * count * count)
        )
        return (center - radius) / denominator

    dependence = {
        "drop_when_left_shuffled": normal
        - metrics["left_shuffled"]["combined_accuracy"],
        "drop_when_right_shuffled": normal
        - metrics["right_shuffled"]["combined_accuracy"],
        "drop_when_both_shuffled": normal - both,
    }
    assertions = {
        "normal_same_song_above_chance_95": wilson_lower(normal, examples) > 0.5,
        "both_context_shuffle_drop_at_least_5pp": normal - both >= 0.05,
        "both_context_shuffle_near_chance": both <= 0.55,
        "left_context_contributes": dependence["drop_when_left_shuffled"] > 0,
        "right_context_contributes": dependence["drop_when_right_shuffled"] > 0,
    }
    report = {
        "verdict": "CONTEXT_DEPENDENCE_PASS" if all(assertions.values()) else "CONTEXT_DEPENDENCE_FAIL",
        "split": args.split,
        "examples": examples,
        "skipped_single_song_batches": skipped_single_song_batches,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
        "dependence": dependence,
        "assertions": assertions,
        "interpretation": (
            "The same plausible donor context is applied to positive and negative targets; "
            "a drop therefore audits context dependence rather than target typicality."
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["verdict"] == "CONTEXT_DEPENDENCE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
