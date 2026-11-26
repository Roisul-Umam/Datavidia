"""Contextual calibration: measure each VLM's built-in answer bias on content-free inputs.

v1 removed bias by subtracting each class's mean log-prob over the dataset. That silently
assumes the six classes are equally frequent: a class the data really contains a lot of
(Fire?) gets pushed down, and its pairs drift to another class (they went to Volcano).
Contextual calibration (Zhao et al., 2021) instead asks the same question about images that
contain no scene at all (flat grey/black/white, noise, and fully blurred dataset tiles that
keep only the global colour). Whatever the model answers there is pure bias; subtracting it
does not depend on how often each class occurs in the data.

writes exp/cache/contentfree_<tag>.npz: lp (n_inputs, 6) log-probs, names
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402

vlm = importlib.import_module("05_vlm_zeroshot")


def inputs(get_pre, get_post, res=448, n_blur=12):
    rng = np.random.default_rng(0)
    flat = lambda v: Image.fromarray(np.full((res, res, 3), v, np.uint8))
    out = {"grey": (flat(128), flat(128)), "black": (flat(10), flat(10)), "white": (flat(245), flat(245))}
    noise = lambda: Image.fromarray(rng.integers(0, 256, (res, res, 3), dtype=np.uint8))
    out["noise"] = (noise(), noise())
    for j, i in enumerate(rng.choice(3180, n_blur, replace=False)):
        blur = lambda x: Image.fromarray(cv2.GaussianBlur(cv2.resize(x, (res, res)), (0, 0), 60))
        out[f"blur{j}"] = (blur(get_pre(i)), blur(get_post(i)))
    return out


def main(model_id, tag, quant, src):
    proc, model = vlm.load(model_id, quant)
    letter_ids, yn = vlm.token_ids(proc)
    get_pre, get_post = vlm.load_src(src, True)
    ins = inputs(get_pre, get_post)
    lps = []
    for name, (a, b) in ins.items():
        if src == "gray":
            a, b = a.convert("L").convert("RGB"), b.convert("L").convert("RGB")
        lp, _ = vlm.score_pair(proc, model, letter_ids, yn, a, b, vlm.orders(3), damage=False)
        lps.append(lp)
        p = np.exp(lp) / np.exp(lp).sum()
        print(f"{name:6s}", " ".join(f"{c[:4]}{v:.2f}" for c, v in zip(vlm.CLASSES, p)), flush=True)
    np.savez(CACHE / f"contentfree_{tag}.npz", lp=np.array(lps), names=np.array(list(ins)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
