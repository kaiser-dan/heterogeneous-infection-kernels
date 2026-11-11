"""Snakemake script: communities of the WeChat friend network from a degree-corrected SBM, and how
kernel membership lines up with them (rule wechat_sbm). Runs in workflow/envs/graph-tool.yaml.

Largest connected component only. Flat DC-SBM by minimum description length (graph-tool
minimize_blockmodel_dl), `inits` seeded restarts, smallest description length kept, blocks
relabelled by size. Per block: mean P(low class) from the K = 2 fit, against a null that permutes
P(low class) over nodes (`n_perm` permutations, 5-95% band).
"""

import graph_tool.all as gt
import numpy as np

edges = np.loadtxt(snakemake.input.edgelist, dtype=int)
p_low = np.load(snakemake.input.fit)["post_prob"][:, 0]
seed = int(snakemake.params.seed)

g = gt.Graph(directed=False)
g.add_vertex(len(p_low))
g.add_edge_list(edges)
lcc = gt.label_largest_component(g)
nodes = np.flatnonzero(lcc.a)
g = gt.Graph(gt.GraphView(g, vfilt=lcc), prune=True)  # keeps vertex order, so vertex i = nodes[i]

fits = []
for i in range(int(snakemake.params.inits)):
    gt.seed_rng(seed + i)
    np.random.seed(seed + i)
    state = gt.minimize_blockmodel_dl(g, state_args={"deg_corr": True})
    fits.append((state.entropy(), np.array(state.get_blocks().a)))  # copy: .a is a view into graph-tool memory
dl, block = min(fits, key=lambda f: f[0])

_, block = np.unique(block, return_inverse=True)
block = np.argsort(np.argsort(-np.bincount(block), kind="stable"))[block]  # 0 = largest
size = np.bincount(block)

x = p_low[nodes]
means = lambda v: np.bincount(block, weights=v) / size
rng = np.random.default_rng(seed)
null = np.array([means(rng.permutation(x)) for _ in range(int(snakemake.params.n_perm))])

np.savez_compressed(
    snakemake.output[0],
    nodes=nodes, block=block, size=size,
    obs=means(x), null_band=np.quantile(null, [0.05, 0.95], axis=0).T,
    overall=x.mean(), dl=dl,
)
print(f"{len(nodes)} nodes, {len(size)} blocks, description length {dl:.1f}")
