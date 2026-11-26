"""Event-level zero-shot: show the VLM 2x2 mosaics of the most-damaged tiles of a cluster.

A single tile is often ambiguous (intact houses, a field). Tiles of one cluster mostly
come from one event, and the tiles where damage is visible reveal what kind of event it
was. For every cluster we build R mosaics (each: four tiles, before-mosaic + after-mosaic
with matching positions) from the tiles with the highest P(damage visible), and score the
six options exactly like the per-tile run (option-order averaged letter log-probs).

usage: python 14_event_vlm.py <hf_model_id> <tag> <k> [--quant 4bit] [--dmg vlm_q8b_clean512]
writes exp/cache/evlm_<tag>_k<k>.npz: labels (N,), lp (k,6) cluster log-probs
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES  # noqa: E402

spec = importlib.util.spec_from_file_location("vlm", Path(__file__).parent / "05_vlm_zeroshot.py")
vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm)
lm_spec = importlib.util.spec_from_file_location("lm", Path(__file__).parent / "12_label_model.py")
lm = importlib.util.module_from_spec(lm_spec)
lm_spec.loader.exec_module(lm)

EVENT_Q = ("Each image is a 2x2 mosaic of four different places hit by the SAME disaster event. The first mosaic was "
           "taken BEFORE the disaster, the second AFTER it, with each place in the same position in both mosaics. "
           "The images may be rotated, blurred or colour-shifted, and flat single-colour regions are missing data; "
           "ignore those artefacts. Which type of disaster was this event?\n{opts}\nAnswer with the letter only.")


def mosaic(get, idx, tile=336):
    canvas = Image.new("RGB", (2 * tile, 2 * tile))
    for j, i in enumerate(idx[:4]):
        canvas.paste(Image.fromarray(get(i)).resize((tile, tile), Image.BICUBIC), ((j % 2) * tile, (j // 2) * tile))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("tag")
    ap.add_argument("k", type=int)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--dmg", default="vlm_q8b_clean512")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--src", default="clean512")
    a = ap.parse_args()

    X = lm.event_features()
    labels = KMeans(a.k, n_init=10, random_state=0).fit_predict(X)
    dmg = np.load(CACHE / f"{a.dmg}.npz")["dmg"]
    dmg = np.nan_to_num(dmg, nan=np.nanmedian(dmg))
    proc, model = vlm.load(a.model_id, a.quant)
    letter_ids, _ = vlm.token_ids(proc)
    get_pre, get_post = vlm.load_src(a.src, True)
    ords = vlm.orders(3)
    rng = np.random.default_rng(0)
    out = np.zeros((a.k, 6))
    t = time.time()
    for c in range(a.k):
        idx = np.where(labels == c)[0]
        top = idx[np.argsort(-dmg[idx])][: max(4, min(4 * a.reps, len(idx)))]
        acc = np.zeros(6)
        for r in range(a.reps):
            sel = rng.choice(top, 4, replace=len(top) < 4)
            A, B = mosaic(get_pre, sel), mosaic(get_post, sel)
            texts = []
            for o in ords:
                opts = "\n".join(f"{vlm.LETTERS[j]}. {vlm.DESC[CLASSES[cc]]}" for j, cc in enumerate(o))
                texts.append(proc.apply_chat_template(vlm._msgs(EVENT_Q.format(opts=opts)), add_generation_prompt=True))
            import torch
            inp = proc(text=texts, images=[[A, B]] * len(texts), return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                logits = model(**inp, logits_to_keep=1).logits[:, -1].float()
            lp = torch.log_softmax(logits[:, letter_ids], 1).cpu().numpy()
            for ri, o in enumerate(ords):
                for j, cc in enumerate(o):
                    acc[cc] += lp[ri, j] / (len(ords) * a.reps)
        out[c] = acc
        print(c, len(idx), f"{time.time()-t:.0f}s", dict(zip([x[:4] for x in CLASSES], np.round(np.exp(acc), 2))), flush=True)
    np.savez(CACHE / f"evlm_{a.tag}_k{a.k}.npz", labels=labels, lp=out)


if __name__ == "__main__":
    main()
