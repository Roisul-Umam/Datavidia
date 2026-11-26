"""Look for natural 'event islands' in the pair embedding space (UMAP 2-D + HDBSCAN).

Pair feature = whitened DINOv2 (colour + gray) pre & post patch embeddings + CLIP.
writes exp/cache/pairfeat.npy (N, D), exp/cache/umap2.npy, exp/cache/hdb_labels.npy,
       exp/viz/event_map.png (+ montages of the largest islands)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, VIZ, list_ids, montage  # noqa: E402


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def pair_features(spec=(("dinov2l_rgbn", "patch", 128), ("dinov2l_gray", "patch", 128), ("clipl_rgbn", "cls", 64))):
    blocks = []
    for name, key, dim in spec:
        a = l2(np.load(CACHE / f"emb_{name}_pre.npz")[key])
        b = l2(np.load(CACHE / f"emb_{name}_post.npz")[key])
        pca = PCA(dim, whiten=True, random_state=0).fit(np.vstack([a, b]))
        a, b = l2(pca.transform(a)), l2(pca.transform(b))
        blocks.append(l2(a + b))  # location identity: what pre and post share
    return l2(np.hstack(blocks))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.cluster import HDBSCAN

    X = pair_features()
    np.save(CACHE / "pairfeat.npy", X)
    U10 = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=0).fit_transform(X)
    U2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
    np.save(CACHE / "umap2.npy", U2)
    lab = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(U10)
    np.save(CACHE / "hdb_labels.npy", lab)
    k = lab.max() + 1
    print("hdbscan clusters:", k, " noise share:", (lab < 0).mean().round(3))
    print("sizes:", sorted(np.bincount(lab[lab >= 0]).tolist(), reverse=True))
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(U2[lab < 0, 0], U2[lab < 0, 1], s=2, c="#cccccc")
    cmap = plt.get_cmap("tab20")
    for c in range(k):
        m = lab == c
        ax.scatter(U2[m, 0], U2[m, 1], s=3, color=cmap(c % 20))
        ax.text(*np.median(U2[m], 0), str(c), fontsize=9, weight="bold")
    ax.set_title("UMAP of pair embeddings, HDBSCAN islands")
    fig.savefig(VIZ / "event_map.png", dpi=90, bbox_inches="tight")
    ids = list_ids()
    pre = np.load(CACHE / "clean512_pre.npy", mmap_mode="r")
    post = np.load(CACHE / "clean512_post.npy", mmap_mode="r")
    out = VIZ / "hdb"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for c in range(k):
        idx = np.where(lab == c)[0]
        pick = rng.choice(idx, min(8, len(idx)), replace=False)
        montage([(f"h{c} n{len(idx)} {ids[i][:6]}", pre[i], post[i]) for i in pick], out / f"h{c:02d}.jpg", thumb=120)
