from pathlib import Path

import pandas as pd

from src.data_processing.operation_outbreak import (
    compute_exposure_counts_from_timeseries,
    generate_oo_exposure_parquet,
)
from workflow.lib.experiment_plan import build_workflow_context


def test_compute_exposure_counts_modes_from_timeseries():
    timeline = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    states = pd.DataFrame(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        index=timeline,
        columns=["A", "B", "C"],
    )
    adjacency = pd.DataFrame(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    m_surv, n_surv = compute_exposure_counts_from_timeseries(
        states=states,
        adjacency=adjacency,
        kmax=4,
        mn_mode="survival",
    )
    m_inc, n_inc = compute_exposure_counts_from_timeseries(
        states=states,
        adjacency=adjacency,
        kmax=4,
        mn_mode="incidence",
    )

    assert m_surv.loc["A"].tolist() == [1, 0, 0, 0]
    assert n_surv.loc["A"].tolist() == [1, 0, 0, 0]
    assert m_surv.loc["B"].tolist() == [0, 0, 0, 0]
    assert n_surv.loc["B"].tolist() == [1, 1, 1, 0]
    assert m_surv.loc["C"].tolist() == [1, 0, 0, 0]
    assert n_surv.loc["C"].tolist() == [2, 0, 0, 0]

    assert m_inc.loc["A"].tolist() == [1, 0, 0, 0]
    assert n_inc.loc["A"].tolist() == [0, 0, 0, 0]
    assert m_inc.loc["B"].tolist() == [0, 0, 0, 0]
    assert n_inc.loc["B"].tolist() == [0, 0, 1, 0]
    assert m_inc.loc["C"].tolist() == [1, 0, 0, 0]
    assert n_inc.loc["C"].tolist() == [0, 0, 0, 0]


def test_generate_oo_exposure_parquet_writes_expected_outputs(tmp_path: Path):
    csv_path = tmp_path / "oo_histories.csv"
    output_dir = tmp_path / "processed_oo"

    rows = pd.DataFrame(
        [
            {"sim_id": 1, "type": "contact", "user_id": "A", "peer_id": "B", "contact_length": 60_000},
            {"sim_id": 1, "type": "contact", "user_id": "B", "peer_id": "C", "contact_length": 30_000},
            {"sim_id": 1, "type": "infection", "user_id": "A", "peer_id": None, "contact_length": None, "time": 1_704_067_200},
            {"sim_id": 1, "type": "infection", "user_id": "C", "peer_id": None, "contact_length": None, "time": 1_704_070_800},
        ]
    )
    rows.to_csv(csv_path, index=False)

    generate_oo_exposure_parquet(
        histories_path=csv_path,
        sim_id=1,
        output_dir=output_dir,
        freq="1h",
        kmax=4,
        backbone_method=None,
        backbone_threshold=None,
        mn_mode="incidence",
    )

    expected_files = ["exposures.parquet", "edges.parquet", "nodes.parquet", "M.parquet", "N.parquet"]
    for filename in expected_files:
        assert (output_dir / filename).exists(), f"Missing expected OO output: {filename}"

    exposures = pd.read_parquet(output_dir / "exposures.parquet")
    nodes = pd.read_parquet(output_dir / "nodes.parquet")
    m_df = pd.read_parquet(output_dir / "M.parquet")
    n_df = pd.read_parquet(output_dir / "N.parquet")

    assert set(exposures["user_id"]) == {"A", "B", "C"}
    assert set(nodes["user_id"]) == {"A", "B", "C"}
    assert m_df.shape == (3, 4)
    assert n_df.shape == (3, 4)
    assert exposures["adopted"].sum() == 2


def test_workflow_context_normalizes_legacy_oo_data_paths(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "processed" / "oo" / "sim_165_backbone_disparity").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "oo" / "sim_165_backbone_disparity_exposures.parquet").write_text("")
    (tmp_path / "configs" / "default_data_prep.yaml").write_text("source: twitter\n")

    config = {
        "name": "research_sweep_oo",
        "source": "oo",
        "oo": {
            "data_path": "data/processed/oo/sim_165_backbone_disparity_exposures.parquet",
        },
        "sweep": {
            "K": [1],
            "stan_file": ["src/models/model_simple_contagion.stan"],
        },
    }

    context = build_workflow_context(config=config, workflow=None)

    assert context.data_cfg_names == ("sim_165_backbone_disparity",)
