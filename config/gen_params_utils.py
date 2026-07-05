"""Shared helper for the gen_params_<exp>.py generators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "workflow" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from model_select import resolve_model, model_tag


def json_col(obj) -> str:
    """Compact, tab/newline-free JSON encoding safe to store in a TSV cell."""
    return json.dumps(obj, separators=(",", ":"))


def write_frame(df: pd.DataFrame, exp: str, wildcard_columns: list[str], outdir: Path) -> None:
    missing = [c for c in wildcard_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{exp}: wildcard_columns {missing} not in frame columns {list(df.columns)}")

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{exp}.tsv", sep="\t", index=False)
    (outdir / f"{exp}.wildcard_columns.json").write_text(
        json.dumps(wildcard_columns, indent=2) + "\n"
    )


def axis_values(spec: dict) -> list[float]:
    """Expand a {start, step, count} axis spec into its grid values."""
    start, step, count = spec["start"], spec["step"], spec["count"]
    return [round(start + step * i, 4) for i in range(int(count))]