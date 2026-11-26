"""Diagnostic: can the VLM recognise WHERE each event island is? (free-form, no option list)

One disaster event = one place. If the VLM can name the region of an island (from the
undamaged pre image), its world knowledge links that region to the disaster type, which is
far more decisive than judging damage on a blurred 448 px tile.
writes exp/cache/geo_probe.csv  (island, idx, id, view, answer)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import importlib  # noqa: E402

from common import CACHE, list_ids  # noqa: E402

vlm = importlib.import_module("05_vlm_zeroshot")
Q = ("This is an aerial image. In which country and region was it most likely taken? "
     "Answer in the form 'Country, region' and nothing else.")


def main(per_island=10, model_id="Qwen/Qwen3-VL-8B-Instruct", quant="4bit"):
    proc, model = vlm.load(model_id, quant)
    get_pre, get_post = vlm.load_src("clean512", True)
    ids = list_ids()
    lab = np.load(CACHE / "hdb_labels.npy")
    rng = np.random.default_rng(0)
    rows, t = [], time.time()
    for h in range(-1, lab.max() + 1):
        idx = np.where(lab == h)[0]
        pick = rng.choice(idx, min(per_island * (3 if h < 0 else 1), len(idx)), replace=False)
        for i in pick:
            for view, get in (("pre", get_pre),):
                img = Image.fromarray(get(i)).resize((448, 448), Image.BICUBIC)
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
                text = proc.apply_chat_template(msgs, add_generation_prompt=True)
                inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
                ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                rows.append(dict(island=h, idx=int(i), id=ids[i], view=view, answer=ans))
        print(f"h{h}: {len(pick)} done {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "geo_probe.csv", index=False)


if __name__ == "__main__":
    main()
