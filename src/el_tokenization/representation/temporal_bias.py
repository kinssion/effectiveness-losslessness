from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..data.midi_events import TICKS_PER_BAR, TICKS_PER_BEAT


def signed_log_bucket(value: int, *, maximum_bucket: int = 31) -> int:
    if value == 0:
        return 0
    sign = 1 if value > 0 else -1
    magnitude = min(maximum_bucket, 1 + int(math.log2(abs(value))))
    return sign * magnitude


def pairwise_time_features(onsets: Sequence[int]) -> np.ndarray:
    values = np.asarray(onsets, dtype=np.int64)
    delta = values[:, None] - values[None, :]
    features = np.stack(
        (
            np.vectorize(signed_log_bucket)(delta),
            np.vectorize(signed_log_bucket)(delta // TICKS_PER_BEAT),
            np.vectorize(signed_log_bucket)(delta // TICKS_PER_BAR),
            np.sign(delta),
        ),
        axis=-1,
    )
    return features.astype(np.int16)
