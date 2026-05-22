"""
Dashboard-aligned Twitter Figure 3 regression tests.

The definitive pass/fail criterion is the same one used in
notebooks/will/dashboard.qmd:
  - Panel A: pipeline vs transcribed figure values within 1.0 percentage point
  - Panel B: pipeline vs transcribed figure values within 1.0 count
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "twitter" / "full"
TRANSCRIBED_CSV = PROJECT_ROOT / "data" / "derived" / "monsted_fig3_transcribed.csv"

Q_SC = 0.192
Q_CC = 0.18
PANEL_A_THRESHOLD = 1.0
PANEL_B_THRESHOLD = 1.0


def _load_pipeline_sums(metric: str, q_value: float):
    con = duckdb.connect()
    m_path = str(PROCESSED_DIR / f"M_{metric}_q{q_value}.parquet")
    n_path = str(PROCESSED_DIR / f"N_{metric}_q{q_value}.parquet")

    cols_info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{m_path}')").fetchall()
    exposure_cols = [name for name, *_ in cols_info if name != "user_id"]
    sum_stmt = ", ".join([f'SUM("{column}") AS "{column}"' for column in exposure_cols])

    m_sums = con.execute(f"SELECT {sum_stmt} FROM read_parquet('{m_path}')").fetchone()
    n_sums = con.execute(f"SELECT {sum_stmt} FROM read_parquet('{n_path}')").fetchone()
    con.close()

    m_dict = {int(exposure_cols[idx]): float(m_sums[idx]) for idx in range(len(exposure_cols))}
    n_dict = {int(exposure_cols[idx]): float(n_sums[idx]) for idx in range(len(exposure_cols))}
    return m_dict, n_dict


def _comparison_rows(model: str, metric: str, kmax: int, m_dict: dict[int, float], n_dict: dict[int, float], transcribed: pd.DataFrame):
    rows = []
    for k in range(1, kmax + 1):
        panel_a = transcribed[
            (transcribed["panel"] == "A")
            & (transcribed["model"] == model)
            & (transcribed["x"] == k)
        ]
        panel_b = transcribed[
            (transcribed["panel"] == "B")
            & (transcribed["model"] == model)
            & (transcribed["x"] == k)
        ]

        m_value = m_dict.get(k, 0.0)
        n_value = n_dict.get(k, 0.0)
        total = m_value + n_value
        pipe_pct = 100.0 * m_value / total if total > 0 else float("nan")
        pipe_count = m_value

        trans_pct = float(panel_a["observed"].iloc[0]) if not panel_a.empty else float("nan")
        trans_count = float(panel_b["observed"].iloc[0]) if not panel_b.empty else float("nan")

        rows.append(
            {
                "model": model,
                "metric": metric,
                "k": k,
                "pipe_pct": pipe_pct,
                "pipe_count": pipe_count,
                "delta_pct": pipe_pct - trans_pct if pd.notna(trans_pct) else float("nan"),
                "delta_count": pipe_count - trans_count if pd.notna(trans_count) else float("nan"),
            }
        )
    return rows


@pytest.fixture(scope="module")
def comparison_table():
    transcribed = pd.read_csv(TRANSCRIBED_CSV, comment="#")
    sc_m, sc_n = _load_pipeline_sums("k", Q_SC)
    cc_m, cc_n = _load_pipeline_sums("kappa", Q_CC)

    rows = []
    rows.extend(_comparison_rows("SC", "k", 7, sc_m, sc_n, transcribed))
    rows.extend(_comparison_rows("CC", "kappa", 8, cc_m, cc_n, transcribed))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pipeline_sc():
    return _load_pipeline_sums("k", Q_SC)


@pytest.fixture(scope="module")
def pipeline_cc():
    return _load_pipeline_sums("kappa", Q_CC)


def _panel_rows(comparison_table: pd.DataFrame, *, model: str):
    return comparison_table[comparison_table["model"] == model].sort_values("k").reset_index(drop=True)


def test_panel_a_sc_matches_dashboard_threshold(comparison_table):
    rows = _panel_rows(comparison_table, model="SC")
    assert rows["delta_pct"].abs().le(PANEL_A_THRESHOLD).all(), rows[["k", "delta_pct"]].to_string(index=False)


def test_panel_a_cc_matches_dashboard_threshold(comparison_table):
    rows = _panel_rows(comparison_table, model="CC")
    assert rows["delta_pct"].abs().le(PANEL_A_THRESHOLD).all(), rows[["k", "delta_pct"]].to_string(index=False)


def test_panel_b_sc_matches_dashboard_threshold(comparison_table):
    rows = _panel_rows(comparison_table, model="SC")
    assert rows["delta_count"].abs().le(PANEL_B_THRESHOLD).all(), rows[["k", "delta_count"]].to_string(index=False)


def test_panel_b_cc_matches_dashboard_threshold(comparison_table):
    rows = _panel_rows(comparison_table, model="CC")
    assert rows["delta_count"].abs().le(PANEL_B_THRESHOLD).all(), rows[["k", "delta_count"]].to_string(index=False)


def test_sc_probabilities_are_bounded(pipeline_sc):
    m_dict, n_dict = pipeline_sc
    for k in sorted(m_dict):
        total = m_dict[k] + n_dict.get(k, 0.0)
        if total == 0:
            continue
        probability = m_dict[k] / total
        assert 0.0 <= probability <= 1.0, f"SC P(adopt|k={k})={probability:.6f} is out of bounds"


def test_cc_probabilities_are_bounded(pipeline_cc):
    m_dict, n_dict = pipeline_cc
    for k in sorted(m_dict):
        total = m_dict[k] + n_dict.get(k, 0.0)
        if total == 0:
            continue
        probability = m_dict[k] / total
        assert 0.0 <= probability <= 1.0, f"CC P(adopt|k={k})={probability:.6f} is out of bounds"


def test_sc_mn_columns_are_nonnegative(pipeline_sc):
    m_dict, n_dict = pipeline_sc
    assert all(value >= 0.0 for value in m_dict.values())
    assert all(value >= 0.0 for value in n_dict.values())


def test_cc_mn_columns_are_nonnegative(pipeline_cc):
    m_dict, n_dict = pipeline_cc
    assert all(value >= 0.0 for value in m_dict.values())
    assert all(value >= 0.0 for value in n_dict.values())


def test_sc_exposure_mass_exists_for_displayed_bins(pipeline_sc):
    m_dict, n_dict = pipeline_sc
    for k in range(1, 8):
        assert m_dict.get(k, 0.0) + n_dict.get(k, 0.0) > 0.0, f"SC exposure mass at k={k} is zero"


def test_cc_exposure_mass_exists_for_displayed_bins(pipeline_cc):
    m_dict, n_dict = pipeline_cc
    for k in range(1, 9):
        assert m_dict.get(k, 0.0) + n_dict.get(k, 0.0) > 0.0, f"CC exposure mass at k={k} is zero"
