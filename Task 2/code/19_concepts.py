"""Grounded visual concepts: ask the VLM concrete yes/no questions instead of 'which disaster?'.

Error analysis of submission v1 (LB 0.305) showed the six-way question is answered from a
vague impression ('new dark area' -> lava or fire), so flood water and burn scars both
became Volcano. Concrete questions (flood water? burned vegetation? lava/ash? collapsed
buildings? fallen trees? coastline?) are easier to judge on a blurred tile, and the mapping
from concepts to disaster types is explicit domain knowledge (see 20_concept_model.py).

Speed: the two images are encoded once; all questions reuse that KV-cache prefix and are
scored in one batched forward pass (P(yes) from the Yes/No next-token logits).

usage: python 19_concepts.py <tag> [--model ...] [--quant 4bit] [--idx file.npy] [--check 1]
writes exp/cache/concepts_<tag>.npz : logit (N, n_concepts) = log P(yes) - log P(no), names
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402

vlm = importlib.import_module("05_vlm_zeroshot")

PREAMBLE = ("Two aerial images of the same place: the first was taken BEFORE an event, the second AFTER it. "
            "Colours may be shifted (vegetation can look purple, red or blue), and the images may be blurred, rotated "
            "or contain flat single-colour patches of missing data; ignore those artefacts. ")
CONCEPTS = {
    "flood": "In the AFTER image, is land, a field or a neighbourhood covered by flood water that was dry in the BEFORE image?",
    "burn": "In the AFTER image, are there burned areas: blackened or charred vegetation, or houses burned down to their foundations?",
    "lava": "In the AFTER image, is there a lava flow or a thick layer of volcanic ash or mud covering the ground, roads or buildings?",
    "collapse": "In the AFTER image, are there collapsed buildings or piles of rubble among the buildings?",
    "treesdown": "In the AFTER image, have many trees been knocked down, uprooted or stripped of their leaves?",
    "roofs": "In the AFTER image, do house roofs look damaged: torn off, with holes, or covered by blue tarps?",
    "coast": "Is this place directly next to a sea or ocean shoreline?",
    "washed": "In the AFTER image, have buildings near the shore been washed away, leaving debris, sand or mud spread over the land?",
    "city": "Is this a dense urban area with many closely packed, multi-storey buildings?",
    "blast": "In the AFTER image, is there a zone of severe destruction centred on one point, such as a blast site in a port or industrial area?",
    "vegetation": "Is most of the scene natural vegetation, forest or fields rather than buildings?",
    "nodamage": "Do the BEFORE and AFTER images look essentially the same, with no visible disaster damage?",
}
# second batch, written after the v2 error analysis: sharper questions for the two main
# confusions (flood water read as sea -> Tsunami; dark ground read as lava -> Volcano) and
# coarse geography/climate context the VLM can judge from the undamaged scene
CONCEPTS2 = {
    "coastpre": "Looking at the BEFORE image only, is there open sea or ocean water in view (not a river, lake, pond or flood water)?",
    "inlandflood": "In the AFTER image, is there muddy flood water over fields, roads or houses that were dry in the BEFORE image?",
    "usa": "Does this place look like it is in the United States (American-style suburbs, roads, houses or landscape)?",
    "tropical": "Is the vegetation tropical, such as palm trees, banana plants or dense rainforest?",
    "snow": "Is the ground covered by snow in either image?",
    "lavaflow": "In the AFTER image, is there a dark, rough lava flow or a grey blanket of volcanic ash that was not there BEFORE?",
    "burnscar": "In the AFTER image, are trees, shrubs or fields burned (blackened, or brown and leafless) compared with the BEFORE image?",
}
if "--set2" in sys.argv:
    CONCEPTS = CONCEPTS2
    sys.argv.remove("--set2")
NAMES = list(CONCEPTS)


def split_template(proc):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": PREAMBLE + "Before:"}, {"type": "image"},
        {"type": "text", "text": "After:"}, {"type": "image"},
        {"type": "text", "text": "<<Q>>"}]}]
    full = proc.apply_chat_template(msgs, add_generation_prompt=True)
    head, tail = full.split("<<Q>>")
    return head, tail


@torch.no_grad()
def score_cached(proc, model, head, tail, yn, a, b):
    """P(yes) logits for every concept, sharing the image prefix through the KV cache."""
    tok = proc.tokenizer
    inp = proc(text=[head], images=[a, b], return_tensors="pt").to("cuda")
    out = model(**inp, use_cache=True, logits_to_keep=1)
    cache, L = out.past_key_values, inp["input_ids"].shape[1]
    qs = [tok(" " + CONCEPTS[n] + " Answer Yes or No." + tail, add_special_tokens=False).input_ids for n in NAMES]
    ql = max(len(q) for q in qs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([q + [pad] * (ql - len(q)) for q in qs], device="cuda")
    qmask = torch.tensor([[1] * len(q) + [0] * (ql - len(q)) for q in qs], device="cuda")
    B = len(qs)
    cache.batch_repeat_interleave(B)
    am = torch.cat([torch.ones(B, L, dtype=qmask.dtype, device="cuda"), qmask], 1)
    cp = torch.arange(L, L + ql, device="cuda")
    # text after the images: all three M-RoPE axes share position = index + rope_delta of the prefix
    delta = model.model.rope_deltas.to("cuda").view(-1, 1)[:1].expand(B, 1)
    pos = (cp.view(1, -1) + delta).unsqueeze(0).expand(3, B, ql)
    o = model(input_ids=ids, past_key_values=cache, attention_mask=am, position_ids=pos, cache_position=cp,
              use_cache=False)
    last = torch.tensor([len(q) - 1 for q in qs], device="cuda")
    lg = o.logits[torch.arange(len(qs), device="cuda"), last].float()
    return (lg[:, yn[0]] - lg[:, yn[1]]).cpu().numpy()


@torch.no_grad()
def score_full(proc, model, head, tail, yn, a, b, name):
    """Reference implementation without cache (used only to validate score_cached)."""
    text = head + " " + CONCEPTS[name] + " Answer Yes or No." + tail
    inp = proc(text=[text], images=[a, b], return_tensors="pt").to("cuda")
    lg = model(**inp, logits_to_keep=1).logits[0, -1].float()
    return float(lg[yn[0]] - lg[yn[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--src", default="clean512")
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--idx", default=None)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()

    proc, model = vlm.load(a.model, a.quant)
    _, yn = vlm.token_ids(proc)
    head, tail = split_template(proc)
    get_pre, get_post = vlm.load_src(a.src, True)
    img = lambda f, i: Image.fromarray(f(i)).resize((a.res, a.res), Image.BICUBIC)

    if a.check:
        for i in (0, 1, 2):
            c = score_cached(proc, model, head, tail, yn, img(get_pre, i), img(get_post, i))
            f = [score_full(proc, model, head, tail, yn, img(get_pre, i), img(get_post, i), n) for n in NAMES]
            print(f"pair {i}: max |cached - full| = {np.abs(c - np.array(f)).max():.3f}   cached={np.round(c, 2).tolist()}")
        return

    n = 3180
    path = CACHE / f"concepts_{a.tag}.npz"
    out = np.load(path)["logit"].copy() if path.exists() else np.full((n, len(NAMES)), np.nan, np.float32)
    todo = np.load(a.idx) if a.idx else np.arange(n)
    todo = [i for i in todo if np.isnan(out[i, 0])]
    t = time.time()
    for k, i in enumerate(todo):
        out[i] = score_cached(proc, model, head, tail, yn, img(get_pre, i), img(get_post, i))
        if k % 200 == 0 or k == len(todo) - 1:
            np.savez(path, logit=out, names=np.array(NAMES))
            print(k, len(todo), f"{time.time()-t:.0f}s", flush=True)
    np.savez(path, logit=out, names=np.array(NAMES))
    print("done", f"{time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
