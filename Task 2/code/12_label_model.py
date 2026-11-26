"""Label model: turn noisy, biased zero-shot signals into final class decisions.

Signals (each (N,6)): VLM option log-probs (several models/views), CLIP change scores.
  1. calibrate: per signal, subtract each class's mean over all pairs (removes the model's
     global class bias, e.g. 'Volcano for everything'), divide by the global std.
  2. combine: weighted sum of calibrated signals.
  3. graph smoothing: kNN graph on artefact-invariant (grayscale-view) pair embeddings.
     Confidence-weighted propagation F = (1-a) C Z + a S F, normalised by the propagated
     confidence, so tiles with visible damage inform intact tiles of the same event.
  4. decision: softmax(F / T) with per-class biases fitted so the predicted class shares
     match a target prior (Sinkhorn-style), then argmax.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES  # noqa: E402


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def calibrate(S, mask=None):
    m = np.isfinite(S[:, 0]) if mask is None else mask
    mu = S[m].mean(0, keepdims=True)
    Z = S - mu
    return Z / Z[m].std()


def load_signals(names):
    out, dmg = {}, []
    for n in names:
        if n.startswith("vlm_"):
            z = np.load(CACHE / f"{n}.npz")
            out[n] = z["lp"].astype(np.float64)
            dmg.append(z["dmg"])
        elif n.startswith("clip_"):
            v = n.split("_")[1]
            z = np.load(CACHE / f"clipzs_{v}.npz")
            out[n] = (z["post"] - 0.5 * z["pre"]).astype(np.float64)
    d = np.nanmean(np.vstack(dmg), 0) if dmg else None
    return out, d


def noise_covariates():
    """Per-pair variables that describe ONLY the organisers' random corruptions.

    The corruptions were applied at random, so the true class cannot depend on them; any
    dependence of a model's class scores on these variables is bias (e.g. synthetic cloud
    puffs read as 'smoke' -> Explosion) and is regressed out.
    """
    import pandas as pd
    ns = pd.read_csv(CACHE / "noise_survey.csv")
    pre, post = ns[ns.which == "pre"].reset_index(drop=True), ns[ns.which == "post"].reset_index(drop=True)
    cols = {"gray": (pre.chroma_spread < 2).astype(float)}
    for w, d in (("pre", pre), ("post", post)):
        c2 = np.load(CACHE / f"cloud64_{w}.npy").mean((1, 2))
        c1 = np.load(CACHE / f"cloud64v1_{w}.npy").mean((1, 2))
        cols[f"cloud2_{w}"] = np.sqrt(c2)
        cols[f"cloud2_any_{w}"] = (c2 > 0.005).astype(float)
        cols[f"cloud1_{w}"] = np.sqrt(c1)
        cols[f"border_{w}"] = np.sqrt(d.border_frac.values)
        cols[f"imp_{w}"] = (d.impulse_frac.values > 0.002).astype(float)
        cols[f"stripe_{w}"] = np.log1p(d.stripe_row.values)
    cc = CACHE / "colour_covariates.npz"
    if cc.exists():  # share of purple/magenta pixels: hue jitter, almost absent in natural scenes
        z = np.load(cc)
        cols["magenta_pre"], cols["magenta_post"] = np.sqrt(z["pre"][:, 0]), np.sqrt(z["post"][:, 0])
    C = np.column_stack(list(cols.values())).astype(np.float64)
    return (C - C.mean(0)) / C.std(0).clip(1e-9), list(cols)


def reliability():
    """How much a pair's own zero-shot evidence can be trusted: less when clouds hide it."""
    c_pre = np.load(CACHE / "cloud64_pre.npy").mean((1, 2))
    c_post = np.load(CACHE / "cloud64_post.npy").mean((1, 2))
    return np.clip((1 - c_pre) * (1 - c_post) ** 2, 0.1, 1.0)


def residualize(Z, C, coef=None):
    """Remove the part of each class score explained linearly by the corruption covariates."""
    m = np.isfinite(Z[:, 0])
    if coef is None:
        A = np.column_stack([np.ones(m.sum()), C[m]])
        coef = np.linalg.lstsq(A, Z[m], rcond=None)[0][1:]  # (p, 6); intercept stays in the signal
    return Z - C @ coef, coef


def knn_graph(X, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sim = np.exp(-(dist / np.median(dist)) ** 2)
    n = len(X)
    W = sparse.csr_matrix((sim.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(1)).ravel()
    Dinv = sparse.diags(1 / d)
    return Dinv @ W  # row-stochastic


def propagate(P, Z, conf, alpha=0.8, iters=30):
    """Confidence-weighted label propagation of score matrix Z over row-stochastic P."""
    src = conf[:, None] * Z
    F, G = src.copy(), conf.copy()
    for _ in range(iters):
        F = (1 - alpha) * src + alpha * (P @ F)
        G = (1 - alpha) * conf + alpha * (P @ G)
    return F / G[:, None].clip(1e-6)


def fit_bias(F, target, T=0.05, iters=3000):
    """Per-class biases b so that the share of argmax(F + b) matches the target shares.

    Sinkhorn on a low-temperature softmax (T small -> nearly hard assignment): the dual
    variables b are the class 'prices' of an optimal transport from pairs to classes with
    the target class sizes, i.e. each class gets the pairs that are relatively most like it.
    """
    b = np.zeros(F.shape[1])
    for _ in range(iters):
        L = (F + b) / T
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        b += T * (np.log(target) - np.log(P.mean(0).clip(1e-12)))
    return b


def event_features(spec=(("dinov2l_gray", "patch", 128), ("clipl_gray", "patch", 64))):
    from sklearn.decomposition import PCA
    blocks = []
    for name, key, dim in spec:
        a = l2(np.load(CACHE / f"emb_{name}_pre.npz")[key])
        b = l2(np.load(CACHE / f"emb_{name}_post.npz")[key])
        pca = PCA(dim, whiten=True, random_state=0).fit(np.vstack([a, b]))
        blocks.append(l2(l2(pca.transform(a)) + l2(pca.transform(b))))
    return l2(np.hstack(blocks))


def run(signals, weights, X, k=15, alpha=0.8, conf=None, target=None, T=1.0):
    Z = sum(w * calibrate(signals[n]) for n, w in weights.items() if w)
    if conf is None:
        conf = np.ones(len(Z))
    P = knn_graph(X, k)
    F = propagate(P, Z, conf, alpha) if alpha > 0 else Z
    b = fit_bias(F, target, T) if target is not None else np.zeros(6)
    return F + b, F


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else ["clip_rgbn", "clip_gray"]
    sig, dmg = load_signals(names)
    X = event_features()
    F, _ = run(sig, {n: 1.0 for n in names}, X, conf=dmg, target=None)
    pred = F.argmax(1)
    print("pred share:", dict(zip(CLASSES, np.round(np.bincount(pred, minlength=6) / len(pred), 3))))
