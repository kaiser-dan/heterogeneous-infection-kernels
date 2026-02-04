import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# --- Color Palettes ---

# Paul Tol's Muted Palette (Colorblind friendly)
PALETTE_MUTED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", 
    "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
]

# User's Custom Palette
PALETTE_CUSTOM = [
    "#0D3B66", # Dark Blue
    "#DF2935", # Red
    "#99C1B9", # Muted Teal
    "#E2E2E2", # Light Gray
    "#F4D35E"  # Yellow
]

# --- Dimensions (Inches) ---
# Optimized for web/notebook display while being export-responsive
WIDTH_SINGLE = 6.5
WIDTH_DOUBLE = 10.0
HEIGHT_STD = 5.0
DPI_PUB = 300

def set_publication_style(palette=PALETTE_MUTED, use_constrained_layout=False):
    """
    Sets Matplotlib rcParams for professional, publication-ready plots.
    Follows minimalist 'Grammar of Graphics' aesthetics.
    """
    plt.rcParams.update({
        # Colors
        "axes.prop_cycle": plt.cycler(color=palette),
        "image.cmap": "viridis",
        
        # Spines and Grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        
        # Ticks
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Export and Rendering
        "figure.dpi": 100,  # Standard for web display
        "savefig.dpi": DPI_PUB,
        "figure.constrained_layout.use": use_constrained_layout,
        "pdf.fonttype": 42, # Ensure text remains editable
        "ps.fonttype": 42
    })
    print(f"Publication style set with {len(palette)} colors in cycle (constrained_layout={use_constrained_layout}).")

def get_continuous_cmap(colors, name="custom_continuous"):
    """
    Generates a perceptually uniform continuous colormap from a list of colors.
    """
    return LinearSegmentedColormap.from_list(name, colors)

# Pre-defined cmaps using specific variants for nice gradients
CMAP_MUTED = get_continuous_cmap([PALETTE_MUTED[0], PALETTE_MUTED[1], PALETTE_MUTED[5], PALETTE_MUTED[6]], "tol_muted")
CMAP_CUSTOM = get_continuous_cmap([PALETTE_CUSTOM[0], PALETTE_CUSTOM[2], PALETTE_CUSTOM[4]], "user_custom")
CMAP_REINFORCEMENT = get_continuous_cmap(["#f7fbff", PALETTE_CUSTOM[0]], "reinforcement_blue") 
CMAP_COMPLEX = get_continuous_cmap(["#fff5f0", PALETTE_CUSTOM[1]], "complex_red")

if __name__ == "__main__":
    # Quick visual test
    import numpy as np
    set_publication_style()
    plt.figure(figsize=(WIDTH_SINGLE, HEIGHT_STD))
    for i in range(5):
        plt.plot(np.arange(10), np.random.randn(10) + i, label=f"Line {i}")
    plt.title("Style Test (Single Column)")
    plt.xlabel("X-axis (units)")
    plt.ylabel("Y-axis (units)")
    plt.legend()
    plt.show()
