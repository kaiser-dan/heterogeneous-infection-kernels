"""
Twitter Retweet Exposure Analysis

Analyzes the M matrix to create exposure response curves:
1. Count of users who retweeted after k exposures
2. Probability of retweeting at exactly k exposures
3. Cumulative probability of retweeting after at least k exposures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_path = Path("/Users/willthompson/Documents/CSDS/side_projects/heterogeneous-infection-kernels/data/processed/twitter/full")
print("Loading M_k_q1.0.parquet...")

try:
    M_k = pd.read_parquet(data_path / "M_k_q1.0.parquet", engine='pyarrow')
except:
    # Try with fastparquet if pyarrow fails
    M_k = pd.read_parquet(data_path / "M_k_q1.0.parquet", engine='fastparquet')

print(f"M matrix shape: {M_k.shape}")
print(f"Number of users (adopters): {len(M_k)}")
print(f"\nColumns: {M_k.columns.tolist()[:10]}...")  # Show first 10 column names

# Get exposure level columns (all columns except user_id)
exposure_cols = [col for col in M_k.columns if col != 'user_id']
k_values = [int(col) for col in exposure_cols]

# 1. Count of users who retweeted after k exposures
# Sum probabilities across users for each k
user_counts_at_k = M_k[exposure_cols].sum(axis=0).values

count_df = pd.DataFrame({
    'k': k_values,
    'expected_users': user_counts_at_k
})

print(f"\nTotal expected users across all k: {count_df['expected_users'].sum():.1f}")
print(f"\nTop 10 exposure levels by user count:")
print(count_df.nlargest(10, 'expected_users'))

# 2. Probability of retweeting at exactly k exposures
total_retweets = user_counts_at_k.sum()
prob_rt_at_k = user_counts_at_k / total_retweets

prob_df = pd.DataFrame({
    'k': k_values,
    'P(RT at k)': prob_rt_at_k
})

print(f"\nTotal probability: {prob_df['P(RT at k)'].sum():.6f}")
print(f"\nTop 10 exposure levels by retweet probability:")
print(prob_df.nlargest(10, 'P(RT at k)'))

# 3. Cumulative probability
cum_prob_at_least_k = np.array([prob_rt_at_k[i:].sum() for i in range(len(prob_rt_at_k))])

cum_df = pd.DataFrame({
    'k': k_values,
    'P(RT | k >= x)': cum_prob_at_least_k
})

# Create plots
fig = plt.figure(figsize=(18, 12))

# Plot 1: User counts
ax1 = plt.subplot(2, 2, 1)
ax1.bar(count_df['k'], count_df['expected_users'], color='steelblue', alpha=0.7)
ax1.set_xlabel('Number of Exposures (k)', fontsize=12)
ax1.set_ylabel('Expected Number of Users', fontsize=12)
ax1.set_title('Distribution of Retweet Counts by Exposure Level', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.text(0.98, 0.95, f'Total: {total_retweets:.0f} expected retweets',
        transform=ax1.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Probability at k
ax2 = plt.subplot(2, 2, 2)
ax2.bar(prob_df['k'], prob_df['P(RT at k)'], color='coral', alpha=0.7)
ax2.set_xlabel('Number of Exposures (k)', fontsize=12)
ax2.set_ylabel('P(RT at k)', fontsize=12)
ax2.set_title('Probability of Retweeting at Exactly k Exposures', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
mode_k = prob_df.loc[prob_df['P(RT at k)'].idxmax(), 'k']
mode_prob = prob_df['P(RT at k)'].max()
ax2.axvline(mode_k, color='red', linestyle='--', alpha=0.5, label=f'Mode: k={mode_k}')
ax2.legend()

# Plot 3: Cumulative probability
ax3 = plt.subplot(2, 2, 3)
ax3.bar(cum_df['k'], cum_df['P(RT | k >= x)'], color='forestgreen', alpha=0.7)
ax3.set_xlabel('Minimum Number of Exposures (k)', fontsize=12)
ax3.set_ylabel('P(RT | k >= x)', fontsize=12)
ax3.set_title('Cumulative Probability: Retweeting After At Least k Exposures', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax3.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
ax3.legend()

# Plot 4: Zoomed in on low k values
ax4 = plt.subplot(2, 2, 4)
mask = count_df['k'] <= 10
ax4.bar(count_df[mask]['k'], count_df[mask]['expected_users'], color='steelblue', alpha=0.7)
ax4.set_xlabel('Number of Exposures (k)', fontsize=12)
ax4.set_ylabel('Expected Number of Users', fontsize=12)
ax4.set_title('Distribution (Zoomed: k ≤ 10)', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
output_path = Path("/Users/willthompson/Documents/CSDS/side_projects/heterogeneous-infection-kernels/notebooks/will/results")
output_path.mkdir(exist_ok=True)
plt.savefig(output_path / 'twitter_exposure_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path / 'twitter_exposure_analysis.png'}")

plt.show()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal expected retweets: {total_retweets:.1f}")
print(f"Mode (most common k): {mode_k}")
print(f"Probability at mode: {mode_prob:.4f}")

# Mean exposure level (weighted average)
mean_k = np.sum(k_values * prob_rt_at_k)
print(f"\nMean exposure level: {mean_k:.2f}")

# Median (50th percentile)
median_idx = np.where(cum_prob_at_least_k <= 0.5)[0]
if len(median_idx) > 0:
    median_k = k_values[median_idx[0]]
    print(f"Median exposure level: {median_k}")

# 90th percentile
percentile_90_idx = np.where(cum_prob_at_least_k <= 0.1)[0]
if len(percentile_90_idx) > 0:
    k_90 = k_values[percentile_90_idx[0]]
    print(f"90th percentile: {k_90} exposures")

# Concentration at low k
prob_low = prob_df[prob_df['k'] <= 5]['P(RT at k)'].sum()
print(f"\nProbability of RT at k ≤ 5: {prob_low:.4f} ({prob_low*100:.1f}%)")

prob_very_low = prob_df[prob_df['k'] <= 2]['P(RT at k)'].sum()
print(f"Probability of RT at k ≤ 2: {prob_very_low:.4f} ({prob_very_low*100:.1f}%)")

# Save summary data
summary = {
    'total_retweets': float(total_retweets),
    'mode_k': int(mode_k),
    'mode_probability': float(mode_prob),
    'mean_k': float(mean_k),
    'median_k': int(median_k) if len(median_idx) > 0 else None,
    'k_90th_percentile': int(k_90) if len(percentile_90_idx) > 0 else None,
    'prob_k_le_5': float(prob_low),
    'prob_k_le_2': float(prob_very_low)
}

import json
with open(output_path / 'twitter_exposure_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary statistics to: {output_path / 'twitter_exposure_summary.json'}")
