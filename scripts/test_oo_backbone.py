import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.operation_outbreak import contact_adjacency, apply_backbone

histories_path = 'data/raw/operation_outbreak/oo-histories-all.csv'
sim_id = 165

all_events = pd.read_csv(histories_path, low_memory=False)
sim_events = all_events[all_events["sim_id"] == sim_id].copy()
histories = sim_events[sim_events["type"] == "contact"].copy()

adj = contact_adjacency(histories, sim_id)
degrees = (adj > 0).sum(axis=1)
print(f"Original Adj shape: {adj.shape}")
thresholds = [0.05, 0.025, 0.01, 0.005, 0.001]

for t in thresholds:
    print(f"\n--- Threshold {t} ---")
    backbone_adj = apply_backbone(adj, method="disparity", threshold=t)
    edges = (backbone_adj.values > 0).sum()
    degrees = (backbone_adj > 0).sum(axis=1)
    print(f"Edges: {edges}")
    print(f"Avg Degree: {degrees.mean():.2f}")
    print(f"Max Degree: {degrees.max()}")
