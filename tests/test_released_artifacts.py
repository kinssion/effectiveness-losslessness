from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from el_tokenization.data.duplicate_families import audit_components_do_not_cross_splits
from el_tokenization.data.pop1k7 import iter_jsonl
from el_tokenization.reporting.build_figures import build_figures
from el_tokenization.reporting.build_tables import build_tables

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/paper_v1"


def _load(relative: str) -> dict[str, object]:
    return json.loads((ARTIFACTS / relative).read_text(encoding="utf-8"))


def test_paper_ledger_values() -> None:
    ai = _load("aggregate_results/pop1k7_ai_paper_ledger.json")
    observed = {row["arm"]: row["mean_bits_per_declared_fact"] for row in ai["arm_summary"]}  # type: ignore[index]
    expected = {
        "A": 7.116180,
        "B": 6.422231,
        "C": 6.326624,
        "D": 6.268093,
        "E": 6.278414,
        "F": 6.486857,
        "G": 6.396310,
        "H": 6.236382,
        "I": 6.334689,
    }
    assert all(abs(float(observed[arm]) - value) < 5e-7 for arm, value in expected.items())
    for row in ai["per_run"]:  # type: ignore[index]
        assert (
            abs(
                float(row["factor_sum_bits_per_declared_fact"])
                - float(row["paper_bits_per_declared_fact"])
            )
            < 2e-8
        )
        expected_side = 4 if row["arm"] == "G" else 0
        assert row["side_information_bits_per_window"] == expected_side


def test_jk_single_access_and_denominators() -> None:
    result = _load("aggregate_results/pop1k7_jk_paper_ledger.json")
    access = result["test_access"]  # type: ignore[index]
    assert access["test_manifest_accessor_calls"] == 1  # type: ignore[index]
    assert access["test_row_file_reads"] == 1  # type: ignore[index]
    assert access["checkpoint_updates"] == 0  # type: ignore[index]
    assert access["test_time_bpe_fit_updates"] == 0  # type: ignore[index]
    for row in result["per_run"]:  # type: ignore[index]
        assert row["evaluated_window_count"] == 1_979
        assert row["model_native_target_count_including_eos"] == 244_248
        assert row["declared_fact_count"] == 242_269


def test_manifest_counts_and_duplicate_audits() -> None:
    expected = {
        "manifests/pop1k7/source_manifest.jsonl": 1_747,
        "manifests/pop1k7/song_split_v1.jsonl": 1_747,
        "manifests/pop1k7/window_manifest_v1.jsonl": 22_450,
        "manifests/commu/source_manifest.jsonl": 9_299,
        "manifests/commu/song_split_v1.jsonl": 9_299,
    }
    for relative, count in expected.items():
        assert sum(1 for _ in iter_jsonl(ROOT / relative)) == count
    for dataset in ("pop1k7", "commu"):
        rows = list(iter_jsonl(ROOT / f"manifests/{dataset}/duplicate_components.jsonl"))
        audit = audit_components_do_not_cross_splits(rows)
        required_fields = (
            ("lineage_id", "exact_hash", "transposition_invariant_hash")
            if dataset == "commu"
            else (
                "lineage_id",
                "exact_hash",
                "transposition_invariant_hash",
                "rhythm_interval_hash",
            )
        )
        assert all(any(row.get(field) is not None for row in rows) for field in required_fields)
        assert all(audit[field] == 0 for field in required_fields)
        assert audit["rhythm_interval_hash"] == (4 if dataset == "commu" else 0)


def test_key_estimation_manifest_is_auditable() -> None:
    path = ROOT / "manifests/pop1k7/key_estimation_v1.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "song_id",
        "content_sha256",
        "analyzer_1_key",
        "analyzer_2_key",
        "agreement",
        "confidence",
        "accepted",
        "canonical_shift",
        "split",
    }
    assert required <= set(reader.fieldnames or ())
    assert len(rows) == 1_747
    accepted = [row for row in rows if row["accepted"].lower() == "true"]
    assert len(accepted) == 1_263
    assert {int(row["canonical_shift"]) for row in accepted} == set(range(-5, 7))


def test_frozen_k_codebook() -> None:
    tokenizer = ARTIFACTS / "run_metadata/pop1k7_jk/train_only_lossless_bpe.json"
    receipt = _load("run_metadata/pop1k7_jk/bpe_training_receipt.json")
    config = _load("tokenizers/pop1k7_K/bpe_config.json")
    observed = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    assert observed == receipt["tokenizer_sha256"] == config["source_tokenizer_sha256"]
    assert config["vocabulary_size"] == 2_048
    assert config["learned_merge_count"] == 894


def test_paper_outputs_are_artifact_driven(tmp_path: Path) -> None:
    tables = build_tables(ARTIFACTS, tmp_path / "tables")
    figures = build_figures(ARTIFACTS, tmp_path / "figures")
    assert all(Path(path).is_file() for path in (*tables.values(), *figures.values()))
    figure_root = ET.parse(figures["figure2"]).getroot()
    text = " ".join(element.text or "" for element in figure_root.iter())
    assert all(arm in text.split() for arm in "ABCDEFGHI")
    learning_root = ET.parse(figures["learning_dynamics"]).getroot()
    assert len(list(learning_root.iter("{http://www.w3.org/2000/svg}polyline"))) >= 35


def test_no_raw_midi_or_model_weights_in_repository() -> None:
    forbidden = {".mid", ".midi", ".pt", ".pth", ".ckpt", ".safetensors"}
    release_roots = (
        "artifacts",
        "configs",
        "manifests",
        "paper",
        "scripts",
        "src",
        "tests",
        "weights",
    )
    assert not [
        path
        for directory in release_roots
        for path in (ROOT / directory).rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden
    ]


def test_release_identity_and_paper_link() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["authors"] == [
        {
            "family-names": "Wang",
            "given-names": "Yi",
            "orcid": "https://orcid.org/0009-0004-6057-8151",
        }
    ]
    assert citation["url"] == "https://arxiv.org/abs/2608.18025"
    assert (
        citation["repository-code"] == "https://github.com/kinssion/effectiveness-losslessness"
    )
    paper = json.loads((ROOT / "paper/paper_version.json").read_text(encoding="utf-8"))
    assert paper["arxiv_id"] == "2608.18025"
    assert paper["content_baseline"] == "m4l_iclr2027_compact_v3"
    assert paper["repository_url"] == "https://github.com/kinssion/effectiveness-losslessness"
    assert paper["reference_pdf_vendored"] is False
    assert not list((ROOT / "paper").glob("*.pdf"))
