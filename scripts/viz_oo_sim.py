import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_oo_diagnostics(sim_id, backbone="disparity"):
    base_path = Path("data/processed/oo")
    prefix = f"sim_{sim_id}_backbone_{backbone}"
    
    m_path = base_path / f"{prefix}_M.parquet"
    n_path = base_path / f"{prefix}_N.parquet"
    nodes_path = base_path / f"{prefix}_nodes.parquet"
    
    if not m_path.exists() or not n_path.exists():
        print(f"Error: Matrices not found for sim {sim_id}. Run data processing first.")
        return

    # Load Data
    M = pd.read_parquet(m_path)
    N = pd.read_parquet(n_path)
    nodes = pd.read_parquet(nodes_path)
    
    # 1. Plot Infection Time Series
    nodes['infection_time'] = pd.to_numeric(nodes['infection_time'], errors='coerce')
    inf_times = nodes.dropna(subset=['infection_time']).sort_values('infection_time')
    inf_times['cumulative'] = np.arange(1, len(inf_times) + 1)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Time Series
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(inf_times['infection_time'], inf_times['cumulative'], marker='o', linestyle='-', markersize=2)
    ax1.set_title(f"Cumulative Infections (Sim {sim_id})")
    ax1.set_xlabel("Unix Timestamp")
    ax1.set_ylabel("Total Infected")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Exposure Kernel (Empirical)
    ax2 = plt.subplot(2, 2, 2)
    m_sum = M.sum(axis=0)
    n_sum = N.sum(axis=0)
    total = m_sum + n_sum
    
    # Avoid div by zero
    p_adopt = np.zeros_like(m_sum, dtype=float)
    mask = total > 0
    p_adopt[mask] = m_sum[mask] / total[mask]
    
    x = M.columns.astype(int)
    ax2.bar(x, p_adopt, color='teal', alpha=0.7)
    ax2.set_title("Empirical Infection Kernel (M/M+N)")
    ax2.set_xlabel("Number of Infected Neighbors (k)")
    ax2.set_ylabel("P(Infection)")
    ax2.set_xticks(x)
    
    # Subplot 3: N Matrix Heatmap
    ax3 = plt.subplot(2, 2, 3)
    # Filter for nodes that have at least one exposure event to make heatmap readable
    n_filt = N[N.sum(axis=1) > 0].head(50) 
    sns.heatmap(n_filt, cmap="Blues", ax=ax3, cbar_kws={'label': 'Exposure Events'})
    ax3.set_title("N Matrix (Top 50 nodes with exposures)")
    ax3.set_xlabel("k")
    ax3.set_ylabel("User ID")
    
    # Subplot 4: M Matrix Heatmap
    ax4 = plt.subplot(2, 2, 4)
    m_filt = M[M.sum(axis=1) > 0].head(50)
    sns.heatmap(m_filt, cmap="Reds", ax=ax4, cbar_kws={'label': 'Adoption Event'})
    ax4.set_title("M Matrix (Top 50 adopters)")
    ax4.set_xlabel("k")
    ax4.set_ylabel("User ID")
    
    plt.tight_layout()
    out_path = f"oo_sim_{sim_id}_diagnostics.png"
    plt.savefig(out_path, dpi=300)
    print(f"Diagnostics plot saved to {out_path}")

if __name__ == "__main__":
    plot_oo_diagnostics(165)
