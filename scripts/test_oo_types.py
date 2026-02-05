import pandas as pd
import numpy as np
from pathlib import Path
from src.data_processing.operation_outbreak import contact_adjacency, infection_time_series

histories_path = 'data/raw/operation_outbreak/oo-histories-all.csv'
sim_id = 165

all_events = pd.read_csv(histories_path, low_memory=False)
sim_events = all_events[all_events["sim_id"] == sim_id].copy()
histories = sim_events[sim_events["type"] == "contact"].copy()
infections = sim_events[sim_events["type"] == "infection"].copy()

adj = contact_adjacency(histories, sim_id)
print(f"Adj index type: {type(adj.index[0])}")

states = infection_time_series(infections, sim_id, participants=adj.index)
print(f"States sum: {states.values.sum()}")
print(f"Infections count: {len(infections)}")
