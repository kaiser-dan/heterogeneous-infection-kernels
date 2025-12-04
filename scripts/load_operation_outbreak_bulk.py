"""
Bulk-load Operation Outbreak simulations into the contagion database.

Usage:
    python scripts/load_operation_outbreak_bulk.py \
        --db data/contagion.db \
        --oo-data-dir data/Blanchard-Thesis-main/data \
        --weight-type duration \
        --freq 1H
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Local imports
from data_processing.database_utils import insert_operation_outbreak_run
from data_processing.operation_outbreak import (
    compute_exposure_counts,
    contact_adjacency,
    infection_time_series,
)


def load_simulation(
    session,
    sim_id: int,
    histories: pd.DataFrame,
    infections: pd.DataFrame,
    *,
    weight_type: str,
    freq: str,
    directed: bool,
    include_exposures: bool,
    overwrite: bool,
):
    run_id = f"oo_sim_{sim_id}_{weight_type}"

    if not overwrite:
        existing = session.query(type(insert_operation_outbreak_run.__annotations__["return"])).filter_by(run_id=run_id).first()  # type: ignore[index]
        if existing:
            print(f"  [skip] {run_id} already in DB")
            return

    sim_hist = histories[histories["sim_id"] == sim_id]
    sim_inf = infections[infections["sim_id"] == sim_id]
    if sim_hist.empty or sim_inf.empty:
        print(f"  [warn] missing data for sim {sim_id}; histories rows: {len(sim_hist)}, infections rows: {len(sim_inf)}")
        return

    print(f"  [build] adjacency (weight={weight_type}, directed={directed})")
    adjacency = contact_adjacency(
        sim_hist,
        sim_id,
        weight=weight_type,
        directed=directed,
    )
    print(f"    -> {adjacency.shape}, edges {(adjacency > 0).sum().sum()}")

    print(f"  [build] infection time series (freq={freq})")
    ts = infection_time_series(
        sim_inf,
        sim_id,
        freq=freq,
    )
    print(f"    -> {ts.shape}")

    exposure_M = exposure_N = None
    if include_exposures:
        print(f"  [build] exposure counts (kmax=8)")
        exposure_M, exposure_N = compute_exposure_counts(
            infections=sim_inf,
            sim_id=sim_id,
            adjacency=adjacency,
            freq=freq,
            participants=adjacency.index,
            kmax=8,
        )

    print(f"  [insert] run_id={run_id}")
    insert_operation_outbreak_run(
        session,
        run_id=run_id,
        sim_id=sim_id,
        adjacency=adjacency,
        timeseries=ts,
        weight_type=weight_type,
        is_directed=directed,
        description=f"Operation Outbreak sim {sim_id} ({weight_type})",
        exposure_success=exposure_M,
        exposure_failure=exposure_N,
    )
    print(f"  [ok] inserted {run_id}")


def main():
    parser = argparse.ArgumentParser(description="Bulk-load Operation Outbreak simulations into the contagion DB.")
    parser.add_argument("--db", type=Path, default=Path("data/contagion.db"), help="SQLite DB path to write.")
    parser.add_argument(
        "--oo-data-dir",
        type=Path,
        default=Path("data") / "Blanchard-Thesis-main" / "data",
        help="Directory containing oo-histories-all.csv and infections_all.csv.",
    )
    parser.add_argument("--weight-type", choices=["duration", "count"], default="duration", help="Adjacency weighting.")
    parser.add_argument("--freq", default="1H", help="Frequency for infection time series.")
    parser.add_argument("--directed", action="store_true", help="Keep directed contact edges (default: undirected).")
    parser.add_argument("--include-exposures", action="store_true", help="Compute and store exposure matrices.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run_ids if present.")
    args = parser.parse_args()

    histories_path = args.oo_data_dir / "oo-histories-all.csv"
    infections_path = args.oo_data_dir / "infections_all.csv"

    if not histories_path.exists() or not infections_path.exists():
        raise FileNotFoundError(f"Could not find OO CSVs in {args.oo_data_dir}")

    print(f"Loading Operation Outbreak data from {args.oo_data_dir} ...")
    histories = pd.read_csv(histories_path, low_memory=False)
    infections = pd.read_csv(infections_path, low_memory=False)

    sim_ids = sorted(set(histories["sim_id"].unique()) & set(infections["sim_id"].unique()))
    print(f"Found {len(sim_ids)} simulations with both histories and infections.")

    engine = create_engine(f"sqlite:///{args.db}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for sim_id in sim_ids:
            print(f"\n=== Simulation {sim_id} ===")
            load_simulation(
                session,
                sim_id,
                histories,
                infections,
                weight_type=args.weight_type,
                freq=args.freq,
                directed=args.directed,
                include_exposures=args.include_exposures,
                overwrite=args.overwrite,
            )
    finally:
        session.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
