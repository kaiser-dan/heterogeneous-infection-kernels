import matplotlib as mpl
import seaborn as sns

# color styling
def set_colors():
    global complexity_cmap
    global palette
    
    complexity_cmap = sns.color_palette("flare_r", as_cmap=True)
    palette = sns.color_palette("YlGnBu")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=palette)
    

def set_fonts(extra_params={}):
    params = {
        "font.family": "sans-serif",
        # "mathtext.fontset": "cm",
        "legend.fontsize": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 8
    }
    for key, value in extra_params.items():
        params[key] = value
    mpl.rcParams.update(params)