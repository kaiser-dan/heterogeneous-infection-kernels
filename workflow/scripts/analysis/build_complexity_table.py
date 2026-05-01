"""Aggregate all model_comparison.parquet files for a sweep into one table.

Each file lives at results/{cfg}/{source}/{dataset}/{instance_id}/model_comparison.parquet.
Path components are parsed and added as columns so every row is self-describing.
"""
import argparse
from pathlib import Path

import pandas as pd


def build_complexity_table(results_dir: str, output_path: str) -> pd.DataFrame:
    root = Path(results_dir)
    records = []
    for p in sorted(root.rglob("model_comparison.parquet")):
        # path: results/{cfg}/{source}/{dataset}/{instance_id}/model_comparison.parquet
        parts = p.relative_to(root).parts  # (source, dataset, instance_id, filename)
        if len(parts) < 4:
            print(f"Warning: unexpected path depth {p} — skipping.")
            continue
        source, dataset, instance_id = parts[0], parts[1], "/".join(parts[2:-1])
        df = pd.read_parquet(p)
        df.insert(0, "instance_id", instance_id)
        df.insert(0, "dataset", dataset)
        df.insert(0, "source", source)
        records.append(df)

    if not records:
        raise ValueError(f"No model_comparison.parquet files found under {results_dir}")

    table = pd.concat(records, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    print(f"Complexity table: {len(table)} rows → {output_path}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Sweep results directory (e.g. results/my_sweep)")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    args = parser.parse_args()
    build_complexity_table(args.results_dir, args.output)
