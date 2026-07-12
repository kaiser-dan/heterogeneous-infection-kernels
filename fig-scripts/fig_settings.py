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
        "legend.fontsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
        "figure.titlesize": 10
    }
    for key, value in extra_params.items():
        params[key] = value
    mpl.rcParams.update(params)