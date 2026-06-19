"""
Data generating process (DGP). Takes structured params -> (M, N), with a params-keyed cache.

Example:
    {
        "graph": {"type": "sbm", "params": {"n": 256, "avg_degree": 18, "epsilon": 0.9, "delta": 0.0}},
        "gamma": 0.1, "rho": 0.125, "tmax": 1000, "rep": 0,
        "kernels":    [{"type": "simple", "beta": 0.01}, {"type": "simple", "beta": 0.05}],
        "assignment": {"mode": "block_flip", "omega": 0.1},
    }

A single-kernel DGP uses a one-element `kernels` list; `assignment` is then ignored.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# assumes this file is at <repo_root>/workflow/dgp.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import (
    build_graph,
    build_kernel_bank,
    assign_kernels,
    build_initial_state,
    heterogeneous_contagion_process,
    get_activations,
)


def dgp_key(dgp: dict) -> str:
    """Stable short hash of the DGP params. 
    Identical DGP -> identical key -> one shared cache file.
    """
    blob = json.dumps(dgp, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def simulate_MN(dgp: dict):
    """Maps DGP params -> (M, N)."""
    rep = dgp["rep"]

    A, kmax = build_graph(dgp["graph"], seed=rep)
    n = A.shape[0]

    bank      = build_kernel_bank(dgp["kernels"], kmax=kmax)
    n_kernels = len(dgp["kernels"])
    if n_kernels == 1:
        labels = np.zeros(n, dtype=int) # everyone -> kernel 0
    else:
        labels = assign_kernels(
            n, dgp["assignment"], n_kernels=n_kernels, seed=rep)

    c  = np.stack([bank[labels[l]] for l in range(n)])
    x0 = build_initial_state(n, dgp["rho"], seed=rep)
    x  = heterogeneous_contagion_process(
        A, dgp["gamma"], c, x0,
        tmin=0, tmax=dgp["tmax"],
        random_seed=rep,        # deterministic given rep
    )
    M, N = get_activations(np.transpose(x), A, kmax)
    return M, N


def _atomic_write_text(dest: Path, text: str):
    """Atomically write text to dest, safe under concurrency."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp) # clean up if replace didn't consume it


def _atomic_savez(dest: Path, **arrays):
    """Atomically write a (compressed) .npz to dest, safe under concurrency."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as f:       # file object -> numpy does NOT append .npz
            np.savez_compressed(f, **arrays)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_sidecar(dgp: dict, key: str, cache_dir: Path):
    """Record what each hash means: data_<key>.json holding the full DGP dict.

    Idempotent and self-checking: if a sidecar already exists for this key, its
    contents must match.
    """
    side = cache_dir / f"data_{key}.json"
    payload = {"dgp_key": key, "dgp": dgp}
    if side.exists():
        existing = json.loads(side.read_text())
        if existing.get("dgp") != dgp:
            raise RuntimeError(
                f"hash collision on {key}: sidecar DGP differs from this one")
        return
    _atomic_write_text(side, json.dumps(payload, sort_keys=True, indent=2))


def get_MN(dgp: dict, key: str, cache_dir: Path):
    """Return (M, N), simulating once and caching keyed on the DGP params.

    * The cache is what makes every fit/fold on a given DGP share identical data!
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_sidecar(dgp, key, cache_dir)

    fp = cache_dir / f"data_{key}.npz"
    if fp.exists():
        try:
            d = np.load(fp)
            M, N = d["M"], d["N"]
            if M.size > 0 and N.size > 0: # guard against empty/corrupt cache
                return M, N
        except Exception:
            pass # fall through and re-simulate
        # cached file was empty or unreadable: drop it and regenerate
        try:
            os.unlink(fp)
        except FileNotFoundError:
            pass

    M, N = simulate_MN(dgp)
    _atomic_savez(fp, M=M, N=N)
    return M, N


def lookup_dgp(key: str, cache_dir="results/data_cache") -> dict:
    """Resolve a hash back to its full DGP dict via the sidecar."""
    side = Path(cache_dir) / f"data_{key}.json"
    if not side.exists():
        raise FileNotFoundError(f"no sidecar for {key} in {cache_dir}")
    return json.loads(side.read_text())["dgp"]
