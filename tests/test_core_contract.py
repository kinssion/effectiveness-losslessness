from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from el_tokenization.data.duplicate_families import audit_components_do_not_cross_splits
from el_tokenization.data.midi_events import DeclaredEvent, canonical_event_order
from el_tokenization.evaluation.factor_decomposition import factor_sum_equals_total
from el_tokenization.evaluation.predictive_ledger import PredictiveLedgerEntry
from el_tokenization.evaluation.receipts import checkpoint_config_hash_matches
from el_tokenization.evaluation.side_information import tonal_shift_bits
from el_tokenization.representation import PARAMETER_COUNTS, build_representation
from el_tokenization.representation.canonicalization import canonicalize_pitch, restore_pitch
from el_tokenization.representation.exact_events import decode_exact_events, encode_exact_events
from el_tokenization.representation.fixed_pitch_geometry import (
    chromatic_phase,
    fifths_phase,
    phase_is_injective,
)
from el_tokenization.representation.musical_time import musical_time_coordinate
from el_tokenization.representation.pitch_factorization import join_pitch, split_pitch
from el_tokenization.representation.reversible_bpe import fit_reversible_bpe
from el_tokenization.representation.serialized_carrier import (
    deserialize_events,
    serialize_events,
)
from el_tokenization.representation.temporal_bias import pairwise_time_features
from el_tokenization.training.test_firewall import TestAccessFirewall as Firewall

ROOT = Path(__file__).resolve().parents[1]


def _events() -> tuple[DeclaredEvent, ...]:
    return (
        DeclaredEvent("NOTE", 0, 96, 60),
        DeclaredEvent("NOTE", 0, 192, 67),
        DeclaredEvent("REST", 192, 96),
        DeclaredEvent("NOTE", 384, 480, 64),
    )


def test_event_codec_roundtrip() -> None:
    assert decode_exact_events(encode_exact_events(_events())) == _events()


def test_rest_codec_roundtrip() -> None:
    rest = (DeclaredEvent("REST", 24, 72),)
    assert decode_exact_events(encode_exact_events(rest)) == rest


def test_same_onset_order() -> None:
    values = (
        DeclaredEvent("NOTE", 0, 96, 72),
        DeclaredEvent("NOTE", 0, 96, 48),
        DeclaredEvent("NOTE", 0, 96, 60),
    )
    assert [event.pitch for event in canonical_event_order(values)] == [48, 60, 72]


def test_duration_roundtrip() -> None:
    assert deserialize_events(serialize_events(_events())) == _events()


def test_time_coordinate_roundtrip() -> None:
    for onset in (0, 23, 96, 383, 384, 3071):
        assert musical_time_coordinate(onset).to_onset_ticks() == onset


def test_pairwise_time_bias_determinism() -> None:
    first = pairwise_time_features([0, 96, 384, 400])
    second = pairwise_time_features([0, 96, 384, 400])
    assert np.array_equal(first, second)


def test_pitch_class_register_roundtrip() -> None:
    assert all(join_pitch(*split_pitch(pitch)) == pitch for pitch in range(128))


def test_canonicalization_inverse() -> None:
    for pitch, shift in ((60, 3), (72, -5), (1, 6)):
        assert restore_pitch(canonicalize_pitch(pitch, shift), shift) == pitch


def test_side_information_bits() -> None:
    assert tonal_shift_bits(windows=1_500, source_songs=1_263) == 6_000
    assert (
        tonal_shift_bits(windows=1_500, source_songs=1_263, charge="per_source_song") == 5_052
    )


def test_chromatic_phase_injective() -> None:
    assert phase_is_injective(chromatic_phase)


def test_fifths_phase_injective() -> None:
    assert phase_is_injective(fifths_phase)


def test_serialized_carrier_roundtrip() -> None:
    assert deserialize_events(serialize_events(_events())) == _events()


def test_bpe_roundtrip() -> None:
    tokens = serialize_events(_events())
    model = fit_reversible_bpe([tokens, tokens], merge_count=16, split="train")
    assert model.decode(model.encode(tokens)) == tokens


def test_bpe_train_only() -> None:
    with pytest.raises(ValueError, match="train only"):
        fit_reversible_bpe([("A", "B")], merge_count=1, split="test")


def test_eos_in_numerator_not_denominator() -> None:
    without_eos = PredictiveLedgerEntry(2, 1.0, 1.0, 1.0, 1.0)
    with_eos = PredictiveLedgerEntry(2, 1.0, 1.0, 1.0, 1.0, eos_bits=2.0)
    assert without_eos.declared_fact_count == with_eos.declared_fact_count == 2
    assert with_eos.bits_per_declared_fact - without_eos.bits_per_declared_fact == 1.0


def test_factor_sum_equals_total() -> None:
    entry = PredictiveLedgerEntry(
        9, 1.0, 2.0, 3.0, 4.0, eos_bits=0.5, side_information_bits=0.25
    )
    assert factor_sum_equals_total(entry)


def test_duplicate_components_do_not_cross_splits() -> None:
    rows = [
        {"split": "train", "lineage_id": "a", "exact_hash": "x"},
        {"split": "train", "lineage_id": "a", "exact_hash": "x"},
        {"split": "test", "lineage_id": "b", "exact_hash": "y"},
    ]
    assert all(value == 0 for value in audit_components_do_not_cross_splits(rows).values())


def test_test_loader_unavailable_during_training() -> None:
    firewall = Firewall()
    with pytest.raises(PermissionError, match="unavailable"):
        firewall.training_loader("test")


def test_checkpoint_config_hash_match(tmp_path: Path) -> None:
    config = tmp_path / "arm.yaml"
    config.write_text("arm: D\n", encoding="utf-8")
    expected = hashlib.sha256(config.read_bytes()).hexdigest()
    assert checkpoint_config_hash_matches({"config_sha256": expected}, config)
    assert not checkpoint_config_hash_matches({"config_sha256": "0" * 64}, config)


def test_active_parameter_counts() -> None:
    for dataset, arms in (("pop1k7", "ABCDEFGHIJK"), ("commu", "AD")):
        for arm in arms:
            config = ROOT / f"configs/{dataset}/{arm}.yaml"
            assert build_representation(config).active_parameter_count == PARAMETER_COUNTS[arm]
