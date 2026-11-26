"""Shared paths and image preprocessing for Task 2 (disaster type from pre/post aerial pairs)."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Datasets"
IMG_DIR = DATA / "Fenomena"
EXP = ROOT / "exp"
CACHE = EXP / "cache"
VIZ = EXP / "viz"
LOGS = EXP / "logs"
MODELS = ROOT / "models"
for _p in (CACHE, VIZ, LOGS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# Keep every download, temp file and model cache on E: (C: is full on this machine).
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]


def list_ids() -> list[str]:
    import pandas as pd

    return pd.read_csv(DATA / "sample_submission.csv")["Id"].tolist()


def read_rgb(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# Noise components seen in EDA: warp padding (black borders), salt-and-pepper impulses,
# horizontal scan-line stripes, blur, global colour casts / grayscale, synthetic clouds.
# --------------------------------------------------------------------------------------

def valid_mask(rgb: np.ndarray, thr: int = 10) -> np.ndarray:
    """True where the pixel is real imagery, False for the black padding left by warps.

    Padding is near-black and connected to the image border, so flood-fill from the
    border over near-black pixels; dark shadows inside the scene are kept.
    """
    dark = (rgb.max(axis=2) <= thr).astype(np.uint8)
    h, w = dark.shape
    ff = dark.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    seeds += [(x, 0) for x in range(0, w, 32)] + [(x, h - 1) for x in range(0, w, 32)]
    seeds += [(0, y) for y in range(0, h, 32)] + [(w - 1, y) for y in range(0, h, 32)]
    for sx, sy in seeds:
        if ff[sy, sx] == 1:
            cv2.floodFill(ff, mask, (sx, sy), 2)
    pad = ff == 2
    # the interpolated edge of a warp is a few pixels of dark-grey ramp: grow the pad a bit
    pad = cv2.dilate(pad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return ~pad


def remove_impulses(rgb: np.ndarray, thr: int = 40) -> np.ndarray:
    """Replace salt-and-pepper outliers with the 3x3 median; leave other pixels untouched."""
    med = cv2.medianBlur(rgb, 3)
    diff = np.abs(rgb.astype(np.int16) - med.astype(np.int16)).max(axis=2)
    out = rgb.copy()
    hit = diff > thr
    out[hit] = med[hit]
    return out


def destripe(rgb: np.ndarray, valid: np.ndarray, axis: int = 0, ksize: int = 31) -> np.ndarray:
    """Remove row-wise (axis=0) additive stripe offsets.

    Stripe noise is a per-row offset shared by every column; scene content is not.
    High-pass each channel along the stripe-normal direction, take the robust (median)
    row profile over valid pixels, and subtract it.
    """
    x = rgb.astype(np.float32)
    # cv2 ksize is (w, h): row stripes need a vertical blur, column stripes a horizontal one
    k = (1, ksize) if axis == 0 else (ksize, 1)
    out = np.empty_like(x)
    vm = valid.astype(bool)
    for c in range(3):
        ch = x[..., c]
        hp = ch - cv2.blur(ch, k)
        hp_m = np.where(vm, hp, np.nan)
        prof = np.nanmedian(hp_m, axis=1) if axis == 0 else np.nanmedian(hp_m, axis=0)
        prof = np.nan_to_num(prof)
        if axis == 0:
            out[..., c] = ch - prof[:, None]
        else:
            out[..., c] = ch - prof[None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def stripe_score(rgb: np.ndarray, valid: np.ndarray, axis: int = 0) -> float:
    """Std of the median row (axis=0) or column (axis=1) high-pass profile on luminance."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = (1, 31) if axis == 0 else (31, 1)
    hp = g - cv2.blur(g, k)
    hp = np.where(valid, hp, np.nan)
    prof = np.nanmedian(hp, axis=1) if axis == 0 else np.nanmedian(hp, axis=0)
    return float(np.nanstd(prof))


def colour_normalise(rgb: np.ndarray, valid: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """Per-channel robust stretch over valid pixels; cancels global casts and gains."""
    out = np.empty(rgb.shape, np.float32)
    vm = valid.astype(bool)
    for c in range(3):
        ch = rgb[..., c].astype(np.float32)
        v = ch[vm] if vm.any() else ch.ravel()
        a, b = np.percentile(v, [lo, hi])
        out[..., c] = (ch - a) / max(b - a, 1.0)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def cloud_mask(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bright, unsaturated, smooth regions (synthetic or real cloud / haze)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v, s = hsv[..., 2] / 255.0, hsv[..., 1] / 255.0
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu * mu, 0))
    m = (v > 0.72) & (s < 0.18) & (sd < 12) & valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return m > 0


def montage(pairs, path, title_fn=None, thumb: int = 160, cols: int = 4):
    """Save a grid of pre|post thumbnails. `pairs` = list of (label, pre_rgb, post_rgb)."""
    rows = (len(pairs) + cols - 1) // cols
    W, H = cols * (2 * thumb + 8), rows * (thumb + 16)
    canvas = np.full((H, W, 3), 255, np.uint8)
    for k, (lab, a, b) in enumerate(pairs):
        r, c = divmod(k, cols)
        x, y = c * (2 * thumb + 8), r * (thumb + 16)
        canvas[y + 14:y + 14 + thumb, x:x + thumb] = cv2.resize(np.asarray(a), (thumb, thumb), interpolation=cv2.INTER_AREA)
        canvas[y + 14:y + 14 + thumb, x + thumb:x + 2 * thumb] = cv2.resize(np.asarray(b), (thumb, thumb), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, str(lab)[:40], (x + 2, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def clean_image(rgb: np.ndarray, size: int = 512, normalise_colour: bool = True):
    """Full cleaning chain. Returns (cleaned uint8 RGB at `size`, valid mask, cloud mask)."""
    valid = valid_mask(rgb)
    x = remove_impulses(rgb)
    x = destripe(x, valid, axis=0)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    if normalise_colour:
        x = colour_normalise(x, vm)
    cm = cloud_mask(x, vm)
    # fill padding with the mean valid colour so models do not see hard black wedges
    if (~vm).any() and vm.any():
        fill = x[vm].mean(axis=0).astype(np.uint8)
        x = x.copy()
        x[~vm] = fill
    return x, vm, cm
