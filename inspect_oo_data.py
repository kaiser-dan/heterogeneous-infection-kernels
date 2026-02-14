import pandas as pd
from pathlib import Path
import numpy as np

# Path to the processed FILE
target_file = Path("data/processed/oo/sim_165_backbone_disparity_exposures.parquet")

if not target_file.exists():
    print(f"File {target_file} does not exist.")
    exit()

print(f"Inspecting File: {target_file}")
try:
    df = pd.read_parquet(target_file)
    print("\nExposures Head (First 5 rows):")
    print(df.head())
    
    if 'A' in df.columns:
        print(f"\nA column sample (row 2): {df.iloc[2]['A']}")
        print(f"k column sample (row 2): {df.iloc[2]['k']}")
    else:
        print("Column 'A' not found!")
        
except Exception as e:
    print(f"Error reading file: {e}")

# Check siblings
parent = target_file.parent
base_name = target_file.name.replace("_exposures.parquet", "")

m_path = parent / f"{base_name}_M.parquet"
n_path = parent / f"{base_name}_N.parquet"

if m_path.exists():
    print(f"\nSibling M found: {m_path}")
    m_df = pd.read_parquet(m_path)
    print(f"M unique values: {np.unique(m_df.values)}")
else:
    print(f"Sibling M NOT found: {m_path}")

if n_path.exists():
    print(f"Sibling N found: {n_path}")
else:
    print(f"Sibling N NOT found: {n_path}")
