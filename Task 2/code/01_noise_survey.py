"""Quantify every noise component on every raw image, and check the cleaning chain removes it.

Output: exp/cache/noise_survey.csv (one row per image: id, which=pre|post, metrics).
"""
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import (CACHE, IMG_DIR, cloud_mask, destripe, list_ids, read_rgb,
                    remove_impulses, stripe_score, valid_mask)


def survey(args):
    iid, which = args
    rgb = read_rgb(IMG_DIR / iid / f"{which}.jpg")
    valid = valid_mask(rgb)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    med = cv2.medianBlur(rgb, 3)
    imp = np.abs(rgb.astype(np.int16) - med.astype(np.int16)).max(axis=2) > 40
    vv = valid
    px = rgb[vv].astype(np.float32) if vv.any() else rgb.reshape(-1, 3).astype(np.float32)
    spread = np.abs(px[:, 0] - px[:, 1]).mean() + np.abs(px[:, 1] - px[:, 2]).mean()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1][vv].mean() if vv.any() else 0
    clean = remove_impulses(rgb)
    s_row_raw = stripe_score(clean, valid, 0)
    s_col_raw = stripe_score(clean, valid, 1)
    ds = destripe(clean, valid, axis=0)
    s_row_after = stripe_score(ds, valid, 0)
    small = cv2.resize(ds, (512, 512), interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    cm = cloud_mask(small, vm)
    lap = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return dict(
        id=iid, which=which,
        border_frac=1 - valid.mean(),
        impulse_frac=imp[vv].mean() if vv.any() else 0,
        stripe_row=s_row_raw, stripe_col=s_col_raw, stripe_row_after=s_row_after,
        chroma_spread=spread, sat_mean=sat,
        r_mean=px[:, 0].mean(), g_mean=px[:, 1].mean(), b_mean=px[:, 2].mean(),
        bright_mean=g[vv].mean() if vv.any() else 0, bright_std=g[vv].std() if vv.any() else 0,
        cloud_frac=cm[vm].mean() if vm.any() else 0,
        sharp_lapvar=float(lap[vm].var()) if vm.any() else 0,
    )


if __name__ == "__main__":
    ids = list_ids()
    jobs = [(i, w) for i in ids for w in ("pre", "post")]
    t = time.time()
    with Pool(12) as pool:
        rows = pool.map(survey, jobs, chunksize=16)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "noise_survey.csv", index=False)
    print(f"done {len(df)} images in {time.time()-t:.0f}s")
    pd.set_option("display.width", 200)
    print(df.groupby("which").describe(percentiles=[.1, .5, .9]).T.to_string())
