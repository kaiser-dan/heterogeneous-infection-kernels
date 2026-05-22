import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data_processing.process_twitter import load_twitter_MN


def _write_summary(tmp_path: Path, rows: list[dict]) -> Path:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    pd.DataFrame(rows).to_parquet(summary_dir / "summary.parquet")
    return summary_dir


def test_load_twitter_MN_returns_expected_dict_keys(tmp_path):
    summary_dir = _write_summary(
        tmp_path,
        [
            {"user_id": "u1", "A_array": np.array([2, 1]), "adopted": True},
            {"user_id": "u2", "A_array": np.array([1, 1]), "adopted": False},
        ],
    )
    config = {"q": 0.5, "exposure_metric": "k", "twitter": {"adopters_only": False, "disable_ipw": True}}
    data = load_twitter_MN(summary_dir, config, kmax=4)
    assert set(data.keys()) == {"M", "N", "row_weights", "kmax_observed", "source_desc"}
    assert data["M"].shape == (2, 4)
    assert data["N"].shape == (2, 4)
    assert data["row_weights"].shape == (2,)


def test_load_twitter_MN_adopters_only_drops_non_adopters(tmp_path):
    summary_dir = _write_summary(
        tmp_path,
        [
            {"user_id": "u1", "A_array": np.array([2, 1]), "adopted": True},
            {"user_id": "u2", "A_array": np.array([1, 1]), "adopted": False},
        ],
    )
    config = {"q": 0.5, "exposure_metric": "k", "twitter": {"adopters_only": True}}
    data = load_twitter_MN(summary_dir, config, kmax=4)
    assert data["M"].shape[0] == 1
    assert np.allclose(data["row_weights"], 1.0)


def test_load_twitter_MN_empty_summary_raises_without_precomputed(tmp_path):
    summary_dir = tmp_path / "empty"
    summary_dir.mkdir()
    config = {"q": 0.5, "exposure_metric": "k"}
    with pytest.raises(ValueError, match="summary.parquet or direct pre-computed"):
        load_twitter_MN(summary_dir, config, kmax=4)
