"""Group-aware split tests."""

from __future__ import annotations

from pathlib import Path

from tclocator.dataset import FieldSample
from tclocator.split import grouped_split, sample_group_id, select_aifs_files

import pandas as pd


def _sample(path: str, init_time: str, lead_hour: int) -> FieldSample:
    """Build one AIFS-like sample for split tests."""

    init = pd.Timestamp(init_time)
    return FieldSample(
        path=Path(path),
        domain_name="aifs",
        valid_time=init + pd.Timedelta(hours=lead_hour),
        lead_hour=lead_hour,
    )


def test_grouped_split_keeps_all_leads_of_init_together() -> None:
    """All leads from one AIFS init cycle must land on the same side."""

    samples = [
        _sample("a_000.pt", "2020-08-01T00:00:00Z", 0),
        _sample("a_006.pt", "2020-08-01T00:00:00Z", 6),
        _sample("a_012.pt", "2020-08-01T00:00:00Z", 12),
        _sample("b_000.pt", "2020-08-02T00:00:00Z", 0),
        _sample("b_006.pt", "2020-08-02T00:00:00Z", 6),
        FieldSample(path=Path("unknown.pt"), domain_name="aifs", valid_time=None, lead_hour=None),
    ]

    train_idx, val_idx, val_groups = grouped_split(samples, group_by="init_time", val_fraction=0.5, seed=7)

    train_groups = {sample_group_id(samples[idx], "init_time") for idx in train_idx}
    val_groups_from_indices = {sample_group_id(samples[idx], "init_time") for idx in val_idx}
    train_groups.discard(None)
    val_groups_from_indices.discard(None)

    assert train_groups.isdisjoint(val_groups_from_indices)
    assert sorted(val_groups_from_indices) == val_groups
    assert len({sample_group_id(samples[idx], "init_time") for idx in [0, 1, 2]}) == 1
    assert ({0, 1, 2}.issubset(train_idx) or {0, 1, 2}.issubset(val_idx))
    assert ({3, 4}.issubset(train_idx) or {3, 4}.issubset(val_idx))
    assert 5 in train_idx


def test_select_aifs_files_uses_same_deterministic_split() -> None:
    """File filtering should preserve whole init cycles."""

    files = [
        Path("AIFS_2020_08_01_00_FCST_0h.pt"),
        Path("AIFS_2020_08_01_00_FCST_6h.pt"),
        Path("AIFS_2020_08_02_00_FCST_0h.pt"),
        Path("AIFS_2020_08_02_00_FCST_6h.pt"),
    ]
    config = {"split": {"group_by": "init_time", "val_fraction": 0.5, "seed": 7}, "seed": 7}

    train_files = select_aifs_files(config, files, "train")
    val_files = select_aifs_files(config, files, "val")

    assert set(train_files).isdisjoint(val_files)
    assert sorted(train_files + val_files) == sorted(files)
    assert len(train_files) == 2
    assert len(val_files) == 2
