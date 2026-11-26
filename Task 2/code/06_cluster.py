"""Group pairs into visual 'events' (same place / same disaster) and save montages for EDA.

usage: python 06_cluster.py <emb_name>[+<emb_name>...] <k> [key]
  e.g. python 06_cluster.py emb_dinov2l_gray+emb_clipl_rgbn 30 both
Writes exp/cache/clusters_<tag>_k<k>.npy and exp/viz/clusters_<tag>_k<k>/cXX.jpg
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, VIZ, list_ids, montage  # noqa: E402


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def load_pair_features(names, key="both", dim=128):
    """Pair feature = [whitened pre, whitened post] per embedding, concatenated across embeddings."""
    blocks = []
    for name in names:
        pre, post = np.load(CACHE / f"{name}_pre.npz"), np.load(CACHE / f"{name}_post.npz")
        if key == "both":
            a = np.hstack([l2(pre["cls"]), l2(pre["patch"])])
            b = np.hstack([l2(post["cls"]), l2(post["patch"])])
        else:
            a, b = l2(pre[key]), l2(post[key])
        pca = PCA(n_components=dim, whiten=True, random_state=0).fit(np.vstack([a, b]))
        a, b = l2(pca.transform(a)), l2(pca.transform(b))
        blocks.append(np.hstack([a, b]) / np.sqrt(2 * len(names)))
    return np.hstack(blocks)


if __name__ == "__main__":
    names = sys.argv[1].split("+")
    k = int(sys.argv[2])
    key = sys.argv[3] if len(sys.argv) > 3 else "both"
    tag = "_".join(n.replace("emb_", "") for n in names) + f"_{key}"
    X = load_pair_features(names, key)
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    np.save(CACHE / f"clusters_{tag}_k{k}.npy", lab)
    print("sizes", np.bincount(lab).tolist())
    ids = list_ids()
    pre = np.load(CACHE / "raw512_pre.npy", mmap_mode="r")
    post = np.load(CACHE / "raw512_post.npy", mmap_mode="r")
    out = VIZ / f"clusters_{tag}_k{k}"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for c in range(k):
        idx = np.where(lab == c)[0]
        pick = rng.choice(idx, size=min(16, len(idx)), replace=False)
        montage([(f"c{c} {ids[i]}", pre[i], post[i]) for i in pick], out / f"c{c:02d}.jpg")
    print("montages in", out)
