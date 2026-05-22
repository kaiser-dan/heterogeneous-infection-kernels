import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("../../data/processed/synthetic")
RESULTS = Path("../../results/simpson_mech_a_20260508")
INST_DIR = "K~1/stan_file~non_parametric_monotonic"

def load_instance_data(data_name, sweep_name):
    M   = pd.read_parquet(DATA / data_name / "M.parquet").values
    N   = pd.read_parquet(DATA / data_name / "N.parquet").values
    return M, N

# Let's find a Mechanism A instance
p = list(RESULTS.glob("synthetic/*/K~1/stan_file~non_parametric_monotonic/kernel_mode_summary.parquet"))[0]
data_name = p.parent.parent.parent.name
M, N = load_instance_data(data_name, "simpson_mech_a_20260508")
print(M.shape)
