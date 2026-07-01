"""
Collect parameter-sweep results, compute kernel complexity per DGP.

Run: uv run python workflow/epsilon_omega_sweep/analyze_complexity_epsilon_omega_sweep.py results/inference_epsilon_omega_sweep

For each DGP (shared dgp_key between the np and simple fits) this:
  1. loads the np kernel posterior draws (mu_np_draws, probability scale),
  2. builds a reference kernel (logit scale),
  3. calls compute_complexity_scores(mu_np_draws, ref),
then averages the resulting complexity over reps per DGP.

Writes:
    complexity_per_dgp.parquet      one row per (DGP, rep)
    complexity_by_setting.parquet   averaged over reps per DGP
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis import (
    compute_complexity_scores,
    posterior_mean_logit,
    simple_reference_logit
)
from src.simulation import (
    simple_kernel
)


EPS = 1e-8
REG = 1e-6


FNAME_RE = re.compile(
    r"^(?P<model>.+?)_epsilon-(?P<epsilon>[0-9.]+)_omega-(?P<omega>[0-9.]+)"
    r"_rep-(?P<rep>[0-9]+)_(?P<key8>[0-9a-f]+)\.npz$"
)


def build_sidecar_index(cache_dir: Path) -> dict:
    """Maps 8-char dgp_key prefix -> full DGP dict."""
    idx = {}
    for side in cache_dir.glob("data_*.json"):
        try:
            payload = json.loads(side.read_text())
        except Exception:
            continue
        key = payload.get("dgp_key", "")
        if key:
            idx[key[:8]] = payload["dgp"]
    return idx


def load_draws(npz_path: Path, name: str):
    try:
        d = np.load(npz_path)
    except Exception:
        return None
    if name not in d.files:
        return None
    return np.asarray(d[name], dtype=float)


def build_reference(simple_path: Path, kmax: int) -> np.ndarray | None:
    """Reference simple-contagion kernel on the LOGIT scale."""
    
    beta = load_draws(simple_path, "draws_beta")
    if beta is None:
        return None
    beta_mean = float(beta.reshape(-1).mean())
    
    return simple_reference_logit(beta_mean, kmax)


def main():
    exp_dir   = Path(sys.argv[1])
    cache_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else exp_dir.parent / "data_cache"
    sidecar   = build_sidecar_index(cache_dir)
    print(f"sidecar index: {len(sidecar)} DGPs from {cache_dir}")

    # group result files by dgp_key prefix: each DGP has an np and a simple fit
    groups: dict[str, dict] = {}
    for fp in sorted(exp_dir.glob("*.npz")):
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        g = groups.setdefault(m.group("key8"), {"files": {}, "fname": m})
        g["files"][m.group("model")] = fp

    rows, skipped = [], []
    # per DGP...
    for key8, g in groups.items():
        files = g["files"]
        if "np" not in files or "simple" not in files:
            skipped.append((key8, f"need np and simple, have {sorted(files)}"))
            continue

        dgp = sidecar.get(key8)
        if dgp is not None:
            epsilon = float(dgp["graph"]["params"]["epsilon"])
            omega   = float(dgp["assignment"]["omega"])
            rep     = int(dgp["rep"])
            src     = "sidecar"
        else:
            mm = g["fname"]
            epsilon = float(mm.group("epsilon"))
            omega   = float(mm.group("omega"))
            rep     = int(mm.group("rep"))
            src     = "filename"

        # load posterior mu draws
        mu_np = load_draws(files["np"], "draws_mu")
        if mu_np is None:
            skipped.append((key8, "np fit missing draws_mu"))
            continue
        if mu_np.ndim == 3:
            mu_np = mu_np[:, 0, :]
            
        # build (simple) reference
        ref = build_reference(files["simple"], kmax=mu_np.shape[1])
        if ref is None:
            skipped.append((key8, "could not build ref (simple fit missing draws_beta)"))
            continue

        try:
            scores = compute_complexity_scores(mu_np, ref, reg=REG)
        except Exception as e:
            skipped.append((key8, f"complexity failed: {e}"))
            continue

        rows.append({"epsilon": epsilon, "omega": omega, "rep": rep,
                     "dgp_key8": key8, "setting_source": src, **scores})

    if not rows:
        print("no usable np/simple pairs found")
        for k, why in skipped[:10]:
            print("  skip", k, why)
        return

    per_dgp = pd.DataFrame(rows)
    per_dgp.to_parquet(exp_dir / "complexity_per_dgp.parquet", index=False)
    print(f"per-DGP: {len(per_dgp)} pairs  ({len(skipped)} skipped)")
    if skipped:
        print(f"  (first few skips: {skipped[:3]})")

    by_setting = (per_dgp.groupby(["epsilon", "omega"], as_index=False)
                         .agg(complexity_mean=("complexity", "mean"),
                              complexity_std=("complexity", "std"),
                              n_reps=("complexity", "size"),
                              mean_distance=("mean_distance", "mean"),
                              erank=("erank", "mean")))
    by_setting.to_parquet(exp_dir / "complexity_by_setting.parquet", index=False)

    piv = by_setting.pivot(index="epsilon", columns="omega", values="complexity_mean")
    print(f"\nmean complexity; rows=epsilon, cols=omega, "
          f"averaged over reps:")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
