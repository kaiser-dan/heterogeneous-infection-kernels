"""Load a committed experiment frame (config/params/<exp>.tsv +
<exp>.wildcard_columns.json) into a Paramspace plus a fast per-job row
lookup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from snakemake.utils import Paramspace


class ExperimentFrame:
    """A committed frame plus its Paramspace, with a typed per-job row lookup.

    `row(wc)` returns the ONE frame row a job's wildcards resolve to 
    with wildcard columns decoded back to their frame dtypes via
    Paramspace.instance() and merged with that row's fixed columns.
    """

    def __init__(self, frame: pd.DataFrame, wildcard_columns: list[str], paramspace: Paramspace):
        self.frame = frame
        self.wildcard_columns = wildcard_columns
        self.paramspace = paramspace
        self._indexed = frame.set_index(wildcard_columns, drop=False)

    def row(self, wc) -> dict:
        inst = self.paramspace.instance(wc)
        if len(self.wildcard_columns) == 1:
            key = inst[self.wildcard_columns[0]]
        else:
            key = tuple(inst[c] for c in self.wildcard_columns)
        return self._indexed.loc[key].to_dict()


def load_experiment(exp: str, params_dir: Path, **paramspace_kwargs) -> ExperimentFrame:
    frame = pd.read_csv(params_dir / f"{exp}.tsv", sep="\t")
    wildcard_columns = json.loads((params_dir / f"{exp}.wildcard_columns.json").read_text())
    paramspace = Paramspace(frame[wildcard_columns], **paramspace_kwargs)
    return ExperimentFrame(frame, wildcard_columns, paramspace)
