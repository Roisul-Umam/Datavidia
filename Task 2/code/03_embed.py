"""Extract rotation-averaged, padding/cloud-masked embeddings from the cleaned cache.

usage: python 03_embed.py <model> <variant>
  model:   dinov2l | clipl
  variant: rgbn (colour-normalised RGB) | gray (luminance only, immune to channel swaps)

Writes exp/cache/emb_<model>_<variant>_<pre|post>.npz with arrays cls (N,D) and patch (N,D).
`patch` is the mean of patch tokens over real, cloud-free cells; `cls` is the class token.
Both are averaged over the four 90-degree rotations (aerial tiles have no canonical up).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402  (also sets HF_HOME)

MODEL, VARIANT = sys.argv[1], sys.argv[2]
DEV = "cuda"
BS = 16


def load_model():
    if MODEL == "dinov2l":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-with-registers-large", torch_dtype=torch.float16)
        mean, std, res, patch = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 448, 14
        n_reg = 4

        def fwd(x):
            h = m(pixel_values=x).last_hidden_state
            return h[:, 0], h[:, 1 + n_reg:]
    elif MODEL == "clipl":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        m = m.half()
        mean, std, res, patch = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758), 336, 14
        v = m.visual
        v.output_tokens = True

        def fwd(x):
            pooled, toks = v(x)
            # project patch tokens into the joint space too, so both live in the CLIP space
            toks = v.ln_post(toks) @ v.proj if toks.shape[-1] != pooled.shape[-1] else toks
            return pooled, toks
    else:
        raise ValueError(MODEL)
    return m.to(DEV).eval(), fwd, torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1), res, patch


def main():
    m, fwd, mean, std, res, patch = load_model()
    mean, std = mean.to(DEV).half(), std.to(DEV).half()
    g = res // patch
    for which in ("pre", "post"):
        src = "clean512" if VARIANT == "rgbn" else "raw512"
        arr = np.load(CACHE / f"{src}_{which}.npy", mmap_mode="r")
        valid = np.load(CACHE / f"valid64_{which}.npy")
        cloud = np.load(CACHE / f"cloud64_{which}.npy")
        n = arr.shape[0]
        cls_out, patch_out = None, None
        t = time.time()
        for s in range(0, n, BS):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + BS])).to(DEV).permute(0, 3, 1, 2).half() / 255
            if VARIANT == "gray":
                y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
                # per-image robust stretch on luminance (cancels brightness/contrast shifts)
                flat = y.flatten(1).float()
                lo = torch.quantile(flat, 0.01, dim=1).view(-1, 1, 1, 1).half()
                hi = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1).half()
                y = ((y - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1)
                x = y.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
            x = (x - mean) / std
            w = torch.from_numpy((valid[s:s + BS] * (1 - cloud[s:s + BS]))).to(DEV)[:, None]
            w = F.adaptive_avg_pool2d(w, (g, g))  # (B,1,g,g) weight per patch
            cls_acc, pat_acc = 0, 0
            with torch.no_grad():
                for r in range(4):
                    xr = torch.rot90(x, r, dims=(2, 3))
                    wr = torch.rot90(w, r, dims=(2, 3)).flatten(1)  # (B, g*g)
                    c, toks = fwd(xr)
                    wr = wr / wr.sum(1, keepdim=True).clamp_min(1e-6)
                    pooled = (toks.float() * wr[..., None]).sum(1)
                    cls_acc = cls_acc + c.float()
                    pat_acc = pat_acc + pooled
            c = (cls_acc / 4).cpu().numpy()
            p = (pat_acc / 4).cpu().numpy()
            if cls_out is None:
                cls_out = np.zeros((n, c.shape[1]), np.float32)
                patch_out = np.zeros((n, p.shape[1]), np.float32)
            cls_out[s:s + BS], patch_out[s:s + BS] = c, p
            if s % (BS * 50) == 0:
                print(which, s, f"{time.time()-t:.0f}s", flush=True)
        np.savez(CACHE / f"emb_{MODEL}_{VARIANT}_{which}.npz", cls=cls_out, patch=patch_out)
        print(which, "done", cls_out.shape, f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
