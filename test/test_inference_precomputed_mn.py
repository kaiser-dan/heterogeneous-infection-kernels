from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.inference import _load_precomputed_twitter_mn


def _write_mn_pair(full_dir: Path, metric: str, q: float, users, m_rows, n_rows):
    full_dir.mkdir(parents=True, exist_ok=True)
    cols = ["user_id", "1", "2", "3"]

    m_df = pd.DataFrame([[u, *m] for u, m in zip(users, m_rows)], columns=cols)
    n_df = pd.DataFrame([[u, *n] for u, n in zip(users, n_rows)], columns=cols)

    m_df.to_parquet(full_dir / f"M_{metric}_q{q}.parquet", index=False)
    n_df.to_parquet(full_dir / f"N_{metric}_q{q}.parquet", index=False)


def test_precomputed_loader_aligns_rows_to_summary_order(tmp_path):
    processed_dir = tmp_path / "processed_twitter"
    full_dir = processed_dir / "full"
    users = ["u1", "u2", "u3"]
    _write_mn_pair(
        full_dir=full_dir,
        metric="k",
        q=0.2,
        users=users,
        m_rows=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        n_rows=[[9, 8, 7], [6, 5, 4], [3, 2, 1]],
    )

    data_cfg_path = tmp_path / "data_cfg.yaml"
    data_cfg_path.write_text(yaml.safe_dump({"processed_dir": str(processed_dir)}))

    # Summary order intentionally permuted.
    summary_df = pd.DataFrame(
        {
            "user_id": ["u3", "u1"],
            "A_array": [np.array([1, 2]), np.array([3])],
            "adopted": [True, False],
        }
    )
    config = {"data_config": str(data_cfg_path)}

    out = _load_precomputed_twitter_mn(
        summary_df=summary_df,
        config=config,
        summary_path=tmp_path / "summary_survival.parquet",
        q=0.2,
        metric="k",
    )
    assert out is not None
    M, N, source_desc = out

    assert str(full_dir) in source_desc
    assert M.shape == (2, 3)
    assert N.shape == (2, 3)

    # Row 0 is u3, row 1 is u1.
    np.testing.assert_allclose(M, np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(N, np.array([[3.0, 2.0, 1.0], [9.0, 8.0, 7.0]]))


def test_precomputed_loader_skips_duplicate_user_rows(tmp_path):
    processed_dir = tmp_path / "processed_twitter"
    full_dir = processed_dir / "full"
    users = ["u1", "u2"]
    _write_mn_pair(
        full_dir=full_dir,
        metric="kappa",
        q=0.18,
        users=users,
        m_rows=[[1, 2, 3], [4, 5, 6]],
        n_rows=[[6, 5, 4], [3, 2, 1]],
    )

    data_cfg_path = tmp_path / "data_cfg.yaml"
    data_cfg_path.write_text(yaml.safe_dump({"processed_dir": str(processed_dir)}))

    summary_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "A_array": [np.array([1]), np.array([2])],
            "adopted": [True, True],
        }
    )
    config = {"data_config": str(data_cfg_path)}

    out = _load_precomputed_twitter_mn(
        summary_df=summary_df,
        config=config,
        summary_path=tmp_path / "summary.parquet",
        q=0.18,
        metric="kappa",
    )
    assert out is None


def test_precomputed_loader_reads_direct_summary_dir_without_summary_parquet(tmp_path):
    summary_dir = tmp_path / "summary.parquet"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "user_id": ["u2", "u1"],
            "1": [0.0, 1.0],
            "2": [2.0, 0.0],
        }
    ).to_parquet(summary_dir / "M.parquet", index=False)
    pd.DataFrame(
        {
            "user_id": ["u2", "u1"],
            "1": [3.0, 4.0],
            "2": [5.0, 6.0],
        }
    ).to_parquet(summary_dir / "N.parquet", index=False)

    out = _load_precomputed_twitter_mn(
        summary_df=pd.DataFrame(),
        config={},
        summary_path=summary_dir,
        q=0.2,
        metric="k",
    )

    assert out is not None
    M, N, source_desc = out
    assert "direct" in source_desc
    np.testing.assert_allclose(M, np.array([[0.0, 2.0], [1.0, 0.0]]))
    np.testing.assert_allclose(N, np.array([[3.0, 5.0], [4.0, 6.0]]))
