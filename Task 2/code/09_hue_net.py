"""Self-supervised hue canonicaliser.

Transform family (covers HSV hue shifts AND all RGB channel permutations exactly):
  code c in [0, 48): refl = c // 24, k = c % 24
  T_c: h -> (refl ? (2/3 - h) : h) + k/24   (mod 1), in HSV with h in [0, 1)
A ResNet-18 is trained to predict c after we apply a random T_c to an image. Images that
were already jittered by the organisers act as label noise spread over all codes, so the
network learns the dominant (natural-colour) mode. Two rounds: round 2 trains on images
corrected by round 1 (cleaner targets).

writes models/hue_net.pt, exp/cache/hue_codes.npz (code, conf per image),
       exp/cache/canon512_{pre,post}.npy (hue-corrected clean512)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, MODELS, VIZ, list_ids, montage  # noqa: E402

DEV = "cuda"
NC = 48


def rgb2hsv(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    d = maxc - minc
    s = d / maxc.clamp_min(1e-6)
    dd = d.clamp_min(1e-6)
    h = torch.where(maxc == r, ((g - b) / dd) % 6, torch.where(maxc == g, (b - r) / dd + 2, (r - g) / dd + 4))
    h = torch.where(d > 1e-6, h / 6, torch.zeros_like(h)) % 1.0
    return h, s, maxc


def hsv2rgb(h, s, v):
    h6 = (h % 1.0) * 6
    i = torch.floor(h6)
    f = h6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], 1).gather(1, i[:, None]).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], 1).gather(1, i[:, None]).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], 1).gather(1, i[:, None]).squeeze(1)
    return torch.stack([r, g, b], 1)


def apply_code(x, c, inverse=False):
    """x (B,3,H,W) in [0,1]; c (B,) long codes."""
    h, s, v = rgb2hsv(x)
    refl = (c // 24).float().view(-1, 1, 1)
    k = (c % 24).float().view(-1, 1, 1) / 24
    if not inverse:
        h = torch.where(refl > 0, (2 / 3 - h) % 1.0, h)
        h = (h + k) % 1.0
    else:
        h = (h - k) % 1.0
        h = torch.where(refl > 0, (2 / 3 - h) % 1.0, h)
    return hsv2rgb(h, s, v)


def make_net():
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(512, NC)
    return net.to(DEV)


MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def batch_from(arrs, idx, codes_in=None, train=True):
    """Load images (index list into a flat (which,k) list), optionally undo known codes, random crop."""
    x = torch.from_numpy(np.stack([np.asarray(arrs[w][k]) for w, k in idx])).to(DEV).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    if codes_in is not None:
        x = apply_code(x, codes_in, inverse=True).clamp(0, 1)
    if train:
        i, j = np.random.randint(0, 33, 2)
        x = x[:, :, i:i + 224, j:j + 224]
        if np.random.rand() < 0.5:
            x = x.flip(3)
        x = torch.rot90(x, np.random.randint(4), dims=(2, 3))
    else:
        x = x[:, :, 16:240, 16:240]
    return x


def train_round(arrs, items, prior_codes, epochs=6, bs=64):
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * (len(items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    net.train()
    t, step = time.time(), 0
    for ep in range(epochs):
        perm = np.random.permutation(len(items))
        tot, acc = 0.0, 0.0
        for s in range(0, len(perm) - bs + 1, bs):
            bi = perm[s:s + bs]
            ci = None if prior_codes is None else torch.from_numpy(prior_codes[bi]).to(DEV)
            x = batch_from(arrs, [items[i] for i in bi], ci)
            c = torch.randint(0, NC, (len(bi),), device=DEV)
            x = apply_code(x, c).clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net((x - MEAN) / STD)
                loss = F.cross_entropy(logits.float(), c, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            tot += loss.item()
            acc += (logits.argmax(1) == c).float().mean().item()
        nb = len(perm) // bs
        print(f"  epoch {ep}: loss {tot/nb:.3f} acc {acc/nb:.3f}  {time.time()-t:.0f}s", flush=True)
    return net.eval()


@torch.no_grad()
def predict(net, arrs, items, bs=64):
    """Return (code, prob of code) for each image, averaged over 4 rotations."""
    P = np.zeros((len(items), NC), np.float32)
    for s in range(0, len(items), bs):
        x = batch_from(arrs, items[s:s + bs], train=False)
        p = 0
        for r in range(4):
            with torch.autocast("cuda", dtype=torch.float16):
                p = p + F.softmax(net((torch.rot90(x, r, dims=(2, 3)) - MEAN) / STD).float(), 1)
        P[s:s + bs] = (p / 4).cpu().numpy()
    return P


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    ids = list_ids()
    n = len(ids)
    arrs = {w: np.load(CACHE / f"clean512_{w}.npy", mmap_mode="r") for w in ("pre", "post")}
    hp = np.load(CACHE / "hue_params.npz")
    sat = {"pre": hp["sat_pre"], "post": hp["sat_post"]}
    all_items = [(w, k) for w in ("pre", "post") for k in range(n)]
    colour = np.array([sat[w][k] >= 0.02 for w, k in all_items])
    items = [it for it, c in zip(all_items, colour) if c]
    print("colour images:", len(items), "of", len(all_items))

    codes = None
    for rnd in range(2):
        print(f"round {rnd}")
        net = train_round(arrs, items, codes)
        # predict on the *observed* images: observed ~= T_code(natural). Round 2 was trained
        # on round-1-corrected images (cleaner natural mode) but is applied to raw ones.
        P = predict(net, arrs, items)
        codes = P.argmax(1).astype(np.int64)
        print("  identity share:", np.mean(codes == 0).round(3), " refl share:", np.mean(codes >= 24).round(3),
              " mean conf:", P.max(1).mean().round(3))
    torch.save(net.state_dict(), MODELS / "hue_net.pt")

    code_full = {w: np.zeros(n, np.int64) for w in ("pre", "post")}
    conf_full = {w: np.ones(n, np.float32) for w in ("pre", "post")}
    for (w, k), c, p in zip(items, codes, P.max(1)):
        code_full[w][k], conf_full[w][k] = c, p
    np.savez(CACHE / "hue_codes.npz", code_pre=code_full["pre"], code_post=code_full["post"],
             conf_pre=conf_full["pre"], conf_post=conf_full["post"])
    # write corrected cache at full 512 resolution
    for w in ("pre", "post"):
        out = np.lib.format.open_memmap(CACHE / f"canon512_{w}.npy", "w+", np.uint8, (n, 512, 512, 3))
        for s in range(0, n, 32):
            x = torch.from_numpy(np.ascontiguousarray(arrs[w][s:s + 32])).to(DEV).permute(0, 3, 1, 2).float() / 255
            c = torch.from_numpy(code_full[w][s:s + 32]).to(DEV)
            y = apply_code(x, c, inverse=True).clamp(0, 1)
            out[s:s + 32] = (y * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        out.flush()
    print("rotation hist (deg):", np.bincount(code_full["pre"] % 24 * 15 // 30, minlength=12).tolist())
    can = np.load(CACHE / "canon512_pre.npy", mmap_mode="r")
    moved = np.where(code_full["pre"] != 0)[0]
    pick = np.random.default_rng(1).choice(moved, 16, replace=False)
    montage([(f"{ids[i][:6]} c{code_full['pre'][i]} p{conf_full['pre'][i]:.2f}", arrs["pre"][i], can[i]) for i in pick],
            VIZ / "hue_net_check.jpg", thumb=170)
    print("done")
