"""Label-free quality check for embeddings.

The pre and post image of a pair show the same place, so a good *location/event* embedding
should map them close together even though the organisers corrupted them independently
(colour, warp, clouds, stripes). We measure:
  r1/r10/mrr : retrieve the pre image of pair i from its post image among all 3180 pre images
  same_k20   : share of pairs whose pre and post land in the same KMeans(20) cluster
usage: python 04_embed_eval.py <name> [<name> ...]   (name = emb_<model>_<variant>)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def prep(a, b, whiten=True, dim=256):
    a, b = l2(a), l2(b)
    if whiten:
        pca = PCA(n_components=min(dim, a.shape[1]), whiten=True, random_state=0).fit(np.vstack([a, b]))
        a, b = l2(pca.transform(a)), l2(pca.transform(b))
    return a, b


def evaluate(a, b):
    s = b @ a.T
    rank = (s > s[np.arange(len(s)), np.arange(len(s))][:, None]).sum(1)
    km = KMeans(20, n_init=4, random_state=0).fit(np.vstack([a, b]))
    la, lb = km.labels_[: len(a)], km.labels_[len(a):]
    return dict(r1=(rank == 0).mean(), r10=(rank < 10).mean(), mrr=(1 / (rank + 1)).mean(),
                same_k20=(la == lb).mean())


if __name__ == "__main__":
    for name in sys.argv[1:]:
        pre, post = np.load(CACHE / f"{name}_pre.npz"), np.load(CACHE / f"{name}_post.npz")
        for key in ("cls", "patch", "both"):
            if key == "both":
                a = np.hstack([l2(pre["cls"]), l2(pre["patch"])])
                b = np.hstack([l2(post["cls"]), l2(post["patch"])])
            else:
                a, b = pre[key], post[key]
            for wh in (False, True):
                r = evaluate(*prep(a, b, whiten=wh))
                print(f"{name:22s} {key:6s} whiten={wh!s:5s} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
