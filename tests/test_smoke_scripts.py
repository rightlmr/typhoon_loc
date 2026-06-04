"""Smoke tests for data-related scripts without real data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _tmp_config(tmp_path: Path) -> Path:
    """Create a temporary config for smoke scripts."""

    with (ROOT / "configs" / "pretrain.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"]["output_dir"] = str(tmp_path / "outputs")
    config["paths"]["label_cache_dir"] = str(tmp_path / "label_cache")
    config["paths"]["norm_stats_era5"] = str(tmp_path / "outputs" / "norm_stats_era5.json")
    config["paths"]["norm_stats_aifs"] = str(tmp_path / "outputs" / "norm_stats_aifs.json")
    config["labels"]["mode"] = "ibtracs"
    config["finetune"]["lead_max"] = 24
    config["model"]["base_channels"] = 4
    config["train"]["epochs"] = 1
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def test_data_scripts_synthetic_smoke(tmp_path: Path) -> None:
    """Phase 0, norm stats, and label cache smoke modes run with no real data."""

    config = _tmp_config(tmp_path)
    python_cmd = [sys.executable]
    conda = "conda"
    probe = subprocess.run(
        [conda, "run", "-n", "tc_loc", "python", "-c", "import sys; print(sys.executable)"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if lines:
            python_cmd = [lines[-1]]
    commands = [
        python_cmd + [str(ROOT / "scripts" / "phase0_consistency_and_displacement.py"), "--config", str(config), "--smoke-synthetic"],
        python_cmd
        + [str(ROOT / "scripts" / "compute_norm_stats.py"), "--config", str(config), "--domain", "all", "--smoke-synthetic"],
        python_cmd + [str(ROOT / "scripts" / "build_label_cache.py"), "--config", str(config), "--smoke-synthetic"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr + result.stdout
