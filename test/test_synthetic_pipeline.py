import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _cmdstan_available() -> bool:
    try:
        import cmdstanpy
        return cmdstanpy.cmdstan_version() is not None
    except Exception:
        return False


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(project_root: Path) -> str:
    return str(project_root / ".venv" / "bin" / "python")


def _venv_snakemake(project_root: Path) -> str:
    return str(project_root / ".venv" / "bin" / "snakemake")


@pytest.fixture(scope="module")
def synthetic_module(project_root: Path):
    script_path = project_root / "workflow" / "scripts" / "simulate" / "generate_synthetic_data.py"
    spec = importlib.util.spec_from_file_location("generate_synthetic_data", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_graph_types(synthetic_module):
    sbm_cfg = {"type": "sbm", "params": {"n": 40, "avg_degree": 6, "epsilon": 0.6}}
    er_cfg = {"type": "er", "params": {"n": 40, "avg_degree": 6}}
    zkc_cfg = {"type": "zkc", "params": {}}

    A_sbm = synthetic_module.build_graph(sbm_cfg, seed=7)
    A_er = synthetic_module.build_graph(er_cfg, seed=7)
    A_zkc = synthetic_module.build_graph(zkc_cfg, seed=7)

    for A in (A_sbm, A_er, A_zkc):
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        assert np.array_equal(A, A.T)

    assert A_sbm.shape == (40, 40)
    assert A_er.shape == (40, 40)
    assert A_zkc.shape == (34, 34)


def test_build_kernel_bank_and_assignments(synthetic_module):
    kmax = 12
    kernels_cfg = [
        {"type": "simple", "beta": 0.02},
        {"type": "sigmoid", "rho_l": 0.01, "rho_h": 0.4, "w": 1.2, "k0": 4},
        {"type": "nonparametric", "values": [0.01 * (i + 1) for i in range(kmax)]},
    ]
    bank = synthetic_module.build_kernel_bank(kernels_cfg=kernels_cfg, kmax=kmax)

    assert len(bank) == 3
    for kernel in bank:
        assert kernel.shape == (kmax,)
        assert np.all(np.isfinite(kernel))

    labels_iid = synthetic_module.assign_kernels(
        n=100,
        graph_type="sbm",
        assignment_cfg={"mode": "iid", "probs": [0.2, 0.3, 0.5]},
        n_kernels=3,
        seed=11,
    )
    assert labels_iid.shape == (100,)
    assert set(np.unique(labels_iid)).issubset({0, 1, 2})

    labels_block = synthetic_module.assign_kernels(
        n=80,
        graph_type="sbm",
        assignment_cfg={"mode": "block_flip", "omega": 0.1},
        n_kernels=2,
        seed=11,
    )
    assert labels_block.shape == (80,)
    assert set(np.unique(labels_block)).issubset({0, 1})


def test_synthetic_main_writes_artifacts(tmp_path: Path, synthetic_module):
    cfg = {
        "name": "unit_test_synth",
        "source": "synthetic",
        "synthetic": {
            "seed": 17,
            "kmax": 12,
            "graph": {
                "type": "sbm",
                "params": {"n": 36, "avg_degree": 5, "epsilon": 0.7},
            },
            "contagion": {
                "gamma": 0.1,
                "tmax": 50,
                "dt": 1,
                "initial_infected": 0.15,
                "assignment": {"mode": "block_flip", "omega": 0.05},
                "kernels": [
                    {"id": "k1", "type": "simple", "beta": 0.02},
                    {"id": "k2", "type": "simple", "beta": 0.05},
                ],
            },
        },
    }

    cfg_path = tmp_path / "synth.yaml"
    out_dir = tmp_path / "processed"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    synthetic_module.main(config_path=cfg_path, output_dir=out_dir, data_name="unit_dataset")

    expected = [
        "M.parquet",
        "N.parquet",
        "edges.parquet",
        "exposures.parquet",
        "kernels.parquet",
        "metadata.parquet",
        "metadata.json",
    ]
    for filename in expected:
        assert (out_dir / filename).exists(), f"Missing synthetic output: {filename}"

    M = pd.read_parquet(out_dir / "M.parquet")
    N = pd.read_parquet(out_dir / "N.parquet")
    exposures = pd.read_parquet(out_dir / "exposures.parquet")
    metadata = pd.read_parquet(out_dir / "metadata.parquet").iloc[0]

    assert M.shape[0] == 36
    assert N.shape == M.shape
    assert exposures.shape[1] == 36
    assert int(metadata["n"]) == 36
    assert metadata["data_name"] == "unit_dataset"


def test_synthetic_overrides_are_applied(tmp_path: Path, synthetic_module):
    base_cfg = {
        "name": "unit_test_override",
        "source": "synthetic",
        "synthetic": {
            "seed": 19,
            "kmax": 10,
            "graph": {
                "type": "sbm",
                "params": {"n": 30, "avg_degree": 4, "epsilon": 0.9},
            },
            "contagion": {
                "gamma": 0.1,
                "tmax": 30,
                "dt": 1,
                "initial_infected": 0.1,
                "assignment": {"mode": "block_flip", "omega": 0.01},
                "kernels": [
                    {"id": "k1", "type": "simple", "beta": 0.01},
                    {"id": "k2", "type": "simple", "beta": 0.02},
                ],
            },
        },
    }

    overrides = {
        "graph": {"params": {"epsilon": 0.3}},
        "contagion": {
            "kernels": [
                {"id": "k1", "type": "simple", "beta": 0.01},
                {"id": "k2", "type": "simple", "beta": 0.08},
            ]
        },
    }

    cfg_path = tmp_path / "override.yaml"
    out_dir = tmp_path / "override_processed"
    cfg_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    synthetic_module.main(
        config_path=cfg_path,
        output_dir=out_dir,
        data_name="override_dataset",
        synthetic_overrides=overrides,
    )

    meta = pd.read_parquet(out_dir / "metadata.parquet").iloc[0]
    graph_params = json.loads(meta["graph_params_json"])
    kernels = json.loads(meta["kernels_json"])

    assert pytest.approx(graph_params["epsilon"], rel=0, abs=1e-12) == 0.3
    assert pytest.approx(kernels[1]["beta"], rel=0, abs=1e-12) == 0.08


def test_snakemake_sim_sweep_dry_run(project_root: Path, tmp_path: Path):
    cmd = [
        _venv_snakemake(project_root),
        "-n",
        "inference",
        "--configfile",
        "configs/research_sim_sweep_synthetic.yaml",
        "--profile",
        "workflow/profiles/local",
        "--runtime-source-cache-path",
        str(tmp_path / "runtime-source-cache"),
    ]
    proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if proc.returncode != 0:
        pytest.fail(f"Snakemake dry-run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "generate_synthetic_processed" in combined
    assert "generate_synthetic_summary" in combined
    assert "run_inference" in combined


def test_snakemake_synthetic_data_prep_end_to_end(project_root: Path, tmp_path: Path):
    """
    Run the full data-prep portion of the Snakemake workflow with the minimal
    test_synthetic config and verify all expected output parquet files exist.
    This exercises the ``generate_synthetic_processed`` rule without requiring
    CmdStan / Stan inference.
    """
    config_path = project_root / "configs" / "test_synthetic.yaml"
    with open(config_path, encoding="utf-8") as fh:
        test_cfg = yaml.safe_load(fh)
    data_name = test_cfg["synthetic"]["dataset_name"]

    cmd = [
        _venv_snakemake(project_root),
        "all",
        "-c1",
        "--configfile", str(config_path),
        "--snakefile", str(project_root / "Snakefile"),
        "--directory", str(project_root),
        "--runtime-source-cache-path", str(tmp_path / "runtime-source-cache"),
        "--rerun-incomplete",
    ]
    proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if proc.returncode != 0:
        pytest.fail(
            f"Snakemake 'all' (synthetic data prep) failed:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    processed_dir = project_root / "data" / "processed" / "synthetic" / data_name
    expected_files = [
        "M.parquet",
        "N.parquet",
        "edges.parquet",
        "exposures.parquet",
        "kernels.parquet",
        "metadata.parquet",
    ]
    for fname in expected_files:
        fpath = processed_dir / fname
        assert fpath.exists(), f"Missing output file: {fpath}"
        df = pd.read_parquet(fpath)
        assert len(df) > 0, f"Output file is empty: {fpath}"


@pytest.mark.skipif(not _cmdstan_available(), reason="CmdStan not installed")
def test_snakemake_synthetic_inference_end_to_end(project_root: Path, tmp_path: Path):
    """
    Run the full Snakemake inference target with the minimal test_synthetic config
    (1 sim instance, 1 model, 100 warmup + 100 sampling, 1 chain) and verify
    that all expected inference output artifacts are produced.
    Skipped automatically when CmdStan is not installed.
    """
    config_path = project_root / "configs" / "test_synthetic.yaml"
    with open(config_path, encoding="utf-8") as fh:
        test_cfg = yaml.safe_load(fh)
    cfg_name = test_cfg["name"]
    data_name = test_cfg["synthetic"]["dataset_name"]

    cmd = [
        _venv_snakemake(project_root),
        "inference",
        "-c1",
        "--configfile", str(config_path),
        "--snakefile", str(project_root / "Snakefile"),
        "--directory", str(project_root),
        "--runtime-source-cache-path", str(tmp_path / "runtime-source-cache"),
        "--rerun-incomplete",
    ]
    proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    if proc.returncode != 0:
        pytest.fail(
            f"Snakemake 'inference' (small sweep) failed:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    results_dir = project_root / "results" / cfg_name / "synthetic" / data_name
    posterior_files = list(results_dir.rglob("posterior_samples.parquet"))
    summary_files = list(results_dir.rglob("inference_summary.parquet"))

    assert posterior_files, f"No posterior_samples.parquet found under {results_dir}"
    assert summary_files, f"No inference_summary.parquet found under {results_dir}"

    for fpath in posterior_files + summary_files:
        df = pd.read_parquet(fpath)
        assert len(df) > 0, f"Inference output is empty: {fpath}"
