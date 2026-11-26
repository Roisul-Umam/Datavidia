"""Second, independent zero-shot signal: CLIP text-image similarity, change-aware.

score_k(i) = cos(post_i, T_k) - lambda * cos(pre_i, T_k)
where T_k is the mean text embedding of several prompts for class k. Subtracting the pre
image's similarity removes 'what this place always looks like' (e.g. a coastal town looks
'tsunami-like' before any wave arrives) and keeps 'what changed'.
writes exp/cache/clipzs_<variant>.npz with post, pre (N,6) cosine similarities.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES  # noqa: E402

PROMPTS = {
    "Earthquake": ["collapsed buildings and rubble after an earthquake", "a city damaged by an earthquake",
                   "destroyed concrete buildings after a strong earthquake", "earthquake damage in a town"],
    "Fire": ["a burned area after a wildfire", "houses burned down by a wildfire", "scorched land and ash after a fire",
             "burnt forest and destroyed homes after a fire"],
    "Hurricane": ["damage after a hurricane", "roofs blown off and fallen trees after a hurricane",
                  "flooded neighborhood after a hurricane", "storm debris after a tropical cyclone"],
    "Tsunami": ["a coastal area devastated by a tsunami", "buildings washed away by a tsunami",
                "mud and debris on the coast after a tsunami", "flooded coastline after a tsunami wave"],
    "Volcano": ["a lava flow from a volcanic eruption", "volcanic ash covering the land",
                "a volcano eruption destroying a village", "pyroclastic flow deposits after an eruption"],
    "Explosion": ["a port destroyed by a massive explosion", "buildings destroyed by a huge blast",
                  "an explosion crater in an industrial area", "blast damage radiating from an explosion"],
}
TEMPLATES = ["an aerial image of {}.", "a satellite image of {}.", "{}"]


def text_matrix():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14-336")
    m = m.eval()
    T = []
    with torch.no_grad():
        for c in CLASSES:
            txt = [t.format(p) for p in PROMPTS[c] for t in TEMPLATES]
            e = m.encode_text(tok(txt)).float()
            e = e / e.norm(dim=1, keepdim=True)
            e = e.mean(0)
            T.append((e / e.norm()).numpy())
    return np.stack(T)


if __name__ == "__main__":
    T = text_matrix()
    np.save(CACHE / "clip_text_matrix.npy", T)
    for variant in ("rgbn", "gray"):
        pre = np.load(CACHE / f"emb_clipl_{variant}_pre.npz")["cls"]
        post = np.load(CACHE / f"emb_clipl_{variant}_post.npz")["cls"]
        pre = pre / np.linalg.norm(pre, axis=1, keepdims=True)
        post = post / np.linalg.norm(post, axis=1, keepdims=True)
        np.savez(CACHE / f"clipzs_{variant}.npz", pre=pre @ T.T, post=post @ T.T)
        d = post @ T.T - pre @ T.T
        pred = d.argmax(1)
        print(variant, "change-argmax share:", dict(zip(CLASSES, np.round(np.bincount(pred, minlength=6) / len(pred), 3))))
        pred = (post @ T.T).argmax(1)
        print(variant, "post-argmax share:  ", dict(zip(CLASSES, np.round(np.bincount(pred, minlength=6) / len(pred), 3))))
