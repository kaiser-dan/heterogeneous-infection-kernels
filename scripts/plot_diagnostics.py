
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path('results/author_models')
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for i, metric in enumerate(['k', 'kappa']):
    for q in [1.0, 0.2]:
        p = results_dir / f'simple_{metric}_q{q}_results.npz'
        if p.exists():
            data = np.load(p)
            m = data['m']
            n = data['n']
            ks = np.arange(1, len(m) + 1)
            probs = m / (m + n)
            ax[i].plot(ks[:20], probs[:20], marker='o', label=f'q={q}')
    
    ax[i].set_title(f'Retweet Probability ({metric})')
    ax[i].set_xlabel(metric)
    ax[i].set_ylabel('Probability')
    ax[i].legend()
    ax[i].set_yscale('log')

plt.tight_layout()
plt.savefig('results/diagnostics_probs.png')
print("Saved diagnostics_probs.png")
