"""Metrics matching tests."""

from __future__ import annotations

import pandas as pd

from tclocator.metrics import match_predictions


def test_match_predictions_uses_lead_hour_when_available() -> None:
    """Predictions from another lead must not satisfy the same valid time."""

    references = pd.DataFrame(
        [
            {
                "ISO_TIME": "2024-09-01T12:00:00Z",
                "SID": "A",
                "LEAD_HOUR": 0,
                "LAT_TRUE": 10.0,
                "LON_TRUE": 120.0,
                "LAT_FIELD": 10.0,
                "LON_FIELD": 120.0,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "ISO_TIME": "2024-09-01T12:00:00Z",
                "LEAD_HOUR": 6,
                "LAT": 10.0,
                "LON": 120.0,
                "CONF": 0.9,
            }
        ]
    )

    matched = match_predictions(predictions, references)

    assert not bool(matched.loc[0, "matched"])
    assert not bool(matched.loc[0, "hit"])
