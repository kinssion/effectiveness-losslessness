from __future__ import annotations


def target_event_budget(events_per_equivalent_epoch: int, equivalent_epochs: float) -> int:
    if events_per_equivalent_epoch <= 0 or equivalent_epochs <= 0:
        raise ValueError("budget inputs must be positive")
    return round(events_per_equivalent_epoch * equivalent_epochs)
