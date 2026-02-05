import pandas as pd
import numpy as np
from src.data_processing.operation_outbreak import contact_adjacency, apply_backbone

histories_path = 'data/raw/operation_outbreak/oo-histories-all.csv'
sim_id = 165

all_events = pd.read_csv(histories_path, low_memory=False)
sim_events = all_events[all_events["sim_id"] == sim_id].copy()
histories = sim_events[sim_events["type"] == "contact"].copy()

adj = contact_adjacency(histories, sim_id)
print(f"Original Adj shape: {adj.shape}")
print(f"Original edges: {(adj.values > 0).sum()}")

backbone_adj = apply_backbone(adj, method="disparity", threshold=0.05)
print(f"Backbone edges: {(backbone_adj.values > 0).sum()}")

# Let's check a higher threshold
backbone_adj_high = apply_backbone(adj, method="disparity", threshold=0.2)
print(f"Backbone edges (0.2): {(backbone_adj_high.values > 0).sum()}")
