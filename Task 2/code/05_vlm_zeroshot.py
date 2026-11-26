"""Zero-shot disaster-type probabilities from a vision-language model (no labels needed).

For each pair the VLM sees the cleaned pre and post image and a multiple-choice question.
We read the next-token logits of the six option letters (no free-form generation) and
average over several option orderings, which cancels the model's letter/position bias.
A second yes/no question asks whether disaster damage is visible; P(yes) is later used
to weight tiles (tiles with visible damage reveal the event type, intact tiles do not).

usage: python 05_vlm_zeroshot.py <hf_model_id> <tag> [--res 448] [--orders 3] [--limit N]
                                 [--quant none|4bit] [--src canon512|clean512|gray] [--cloudfill 1]
writes exp/cache/vlm_<tag>.npz  lp (N,6) mean log-probs in CLASSES order, dmg (N,) P(damage visible)
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CLASSES  # noqa: E402

DESC = {
    "Earthquake": "Earthquake (ground shaking: collapsed or cracked buildings and rubble spread across a town)",
    "Fire": "Fire (wildfire or urban fire: burned, blackened or grey ash ground, scorched vegetation, houses burned to their foundations)",
    "Hurricane": "Hurricane (tropical storm wind and rain: roofs torn off, blue tarps, fallen or stripped trees, scattered debris, flooding)",
    "Tsunami": "Tsunami (sea wave flooding the coast: buildings washed away, mud and debris over land, changed shoreline)",
    "Volcano": "Volcano (eruption: lava flows, grey volcanic ash or mud covering land, roads and buildings)",
    "Explosion": "Explosion (a large blast: buildings destroyed around one point with damage radiating outwards, often an industrial or port area)",
}
LETTERS = ["A", "B", "C", "D", "E", "F"]
PREAMBLE = ("These are two aerial images of the same location. The first was taken BEFORE a disaster, the second AFTER it. "
            "The images may be rotated, blurred or colour-shifted, and flat single-colour regions are missing data "
            "(removed clouds or borders); ignore all of those artefacts. ")
QUESTION = PREAMBLE + "Which type of disaster most likely affected this area?\n{opts}\nAnswer with the letter only."
DAMAGE_Q = PREAMBLE + ("Comparing the two images, is there clearly visible disaster damage in the AFTER image "
                       "(destroyed, burned, flooded, buried or debris-covered buildings or land)? Answer Yes or No.")


def orders(n):
    rng = np.random.default_rng(0)
    base = [list(range(6)), list(range(5, -1, -1))]
    while len(base) < n:
        base.append(list(rng.permutation(6)))
    return base[:n]


def load(model_id, quant="none"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "left"
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    return proc, model


def _fill_clouds(img, cloud64, valid64):
    """Paint cloud cells with the mean colour of the clear, valid part of the image."""
    m = cv2.resize(cloud64, (512, 512), interpolation=cv2.INTER_LINEAR) > 0.25
    if not m.any():
        return img
    m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ok = (~m) & (cv2.resize(valid64, (512, 512), interpolation=cv2.INTER_LINEAR) > 0.5)
    out = img.copy()
    out[m] = img[ok].mean(0).astype(np.uint8) if ok.any() else 128
    return out


def load_src(src="canon512", cloudfill=True):
    """Return (pre, post) accessors i -> uint8 RGB (512,512,3) for the chosen input variant."""
    base = "clean512" if src == "gray" else src
    arr = {w: np.load(CACHE / f"{base}_{w}.npy", mmap_mode="r") for w in ("pre", "post")}
    cl = {w: np.load(CACHE / f"cloud64_{w}.npy") for w in ("pre", "post")}
    va = {w: np.load(CACHE / f"valid64_{w}.npy") for w in ("pre", "post")}

    def get(w):
        def f(i):
            x = np.asarray(arr[w][i])
            if src == "gray":
                g = (x.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)
                x = np.repeat(g[..., None], 3, 2)
            if cloudfill:
                x = _fill_clouds(x, cl[w][i], va[w][i])
            return x
        return f
    return get("pre"), get("post")


def _msgs(text):
    return [{"role": "user", "content": [
        {"type": "text", "text": "Before:"}, {"type": "image"},
        {"type": "text", "text": "After:"}, {"type": "image"},
        {"type": "text", "text": text}]}]


@torch.no_grad()
def score_pair(proc, model, letter_ids, yn_ids, a, b, ords, damage=True):
    texts = []
    for o in ords:
        opts = "\n".join(f"{LETTERS[j]}. {DESC[CLASSES[c]]}" for j, c in enumerate(o))
        texts.append(proc.apply_chat_template(_msgs(QUESTION.format(opts=opts)), add_generation_prompt=True))
    if damage:
        texts.append(proc.apply_chat_template(_msgs(DAMAGE_Q), add_generation_prompt=True))
    inp = proc(text=texts, images=[[a, b]] * len(texts), return_tensors="pt", padding=True).to("cuda")
    logits = model(**inp, logits_to_keep=1).logits[:, -1].float()
    lp = torch.log_softmax(logits[:len(ords), letter_ids], dim=1).cpu().numpy()
    acc = np.zeros(6)
    for r, o in enumerate(ords):
        for j, c in enumerate(o):
            acc[c] += lp[r, j]
    dmg = float(torch.softmax(logits[-1, yn_ids], 0)[0]) if damage else np.nan
    return acc / len(ords), dmg


def token_ids(proc):
    tok = proc.tokenizer
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    yn_ids = [tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]]
    return letter_ids, yn_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("tag")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--src", default="canon512")
    ap.add_argument("--cloudfill", type=int, default=1)
    ap.add_argument("--idx", default=None, help="npy file with a subset of pair indices")
    a = ap.parse_args()

    proc, model = load(a.model_id, a.quant)
    letter_ids, yn_ids = token_ids(proc)
    get_pre, get_post = load_src(a.src, bool(a.cloudfill))
    n = 3180
    path = CACHE / f"vlm_{a.tag}.npz"
    if path.exists():
        z = np.load(path)
        lp_all, dmg_all = z["lp"].copy(), z["dmg"].copy()
    else:
        lp_all, dmg_all = np.full((n, 6), np.nan, np.float32), np.full(n, np.nan, np.float32)
    todo = np.load(a.idx) if a.idx else np.arange(n if a.limit is None else a.limit)
    todo = [i for i in todo if np.isnan(lp_all[i, 0])]
    ords = orders(a.orders)
    t = time.time()
    for k, i in enumerate(todo):
        pa = Image.fromarray(get_pre(i)).resize((a.res, a.res), Image.BICUBIC)
        pb = Image.fromarray(get_post(i)).resize((a.res, a.res), Image.BICUBIC)
        lp_all[i], dmg_all[i] = score_pair(proc, model, letter_ids, yn_ids, pa, pb, ords)
        if k % 100 == 0 or k == len(todo) - 1:
            np.savez(path, lp=lp_all, dmg=dmg_all)
            print(k, len(todo), f"{time.time()-t:.0f}s", dict(zip([c[:4] for c in CLASSES], np.round(np.exp(lp_all[i]), 2))),
                  f"dmg={dmg_all[i]:.2f}", flush=True)
    np.savez(path, lp=lp_all, dmg=dmg_all)
    print("done", f"{time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
