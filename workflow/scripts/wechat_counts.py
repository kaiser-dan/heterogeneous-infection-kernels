"""Snakemake script: WMdata raw logs -> activation counts and friend edgelist (rule wechat_counts)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.empirical import wechat_MN

M, N, users, edges = wechat_MN(snakemake.input, kmax=int(snakemake.params.kmax))

np.savez_compressed(snakemake.output[0], M=M, N=N, users=users)
np.savetxt(snakemake.output[1], edges, fmt="%d")
print(f"{len(users)} users, {int((M + N).sum())} trials, {int(M.sum())} adoptions, {len(edges)} friend edges")
