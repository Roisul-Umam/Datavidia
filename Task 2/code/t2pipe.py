"""End-to-end inference pipeline for Task 2 (used by the notebook).

Every heavy step writes to exp/cache/ and is skipped when its output already exists, so the
notebook can be re-run in minutes from cache, or from scratch (~4-5 h on an RTX 3060).
No step trains on labels: the backbones are pretrained, the VLMs are used zero-shot, and
the label model's parameters (calibration means/scales, class biases) are loaded from
models/ and applied as-is.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import CACHE, CLASSES, MODELS, ROOT, list_ids  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, HF_HOME=str(ROOT / ".cache" / "hf"), TORCH_HOME=str(ROOT / ".cache" / "torch"),
           TMP=str(ROOT / ".cache" / "tmp"), TEMP=str(ROOT / ".cache" / "tmp"), PYTHONWARNINGS="ignore")


def _run(script, *args):
    """Run one pipeline script, streaming its output into the notebook."""
    cmd = [PY, "-W", "ignore", str(CODE / script), *map(str, args)]
    print("$", " ".join(cmd[3:]), flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Loading weights" in line or "Warning" in line or "warn(" in line:
            continue
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"{script} failed")


def have(*names):
    return all((CACHE / n).exists() for n in names)


def step_clean():
    if not have("noise_survey.csv"):
        _run("01_noise_survey.py")
    if not have("clean512_pre.npy", "raw512_post.npy", "valid64_post.npy"):
        _run("02_clean_cache.py")
    if not have("cloud64v1_post.npy"):
        _run("10_cloud_v2.py")
    if not have("colour_covariates.npz"):
        _run("16_colour_covariates.py")


def step_embed():
    for model, variant in [("dinov2l", "gray"), ("dinov2l", "rgbn"), ("clipl", "rgbn"), ("clipl", "gray")]:
        if not have(f"emb_{model}_{variant}_post.npz"):
            _run("03_embed.py", model, variant)
    if not have("clipzs_gray.npz"):
        _run("07_clip_zeroshot.py")


def step_vlm(cfg):
    for v in cfg["tile_vlm"]:
        f = CACHE / f"vlm_{v['tag']}.npz"
        if not f.exists() or not np.isfinite(np.load(f)["lp"][:, 0]).all():
            _run("05_vlm_zeroshot.py", v["model"], v["tag"], "--src", v["src"], "--quant", v["quant"])
    for e in cfg.get("event_vlm", []):
        if not have(f"evlm_{e['tag']}_k{e['k']}.npz"):
            _run("14_event_vlm.py", e["model"], e["tag"], e["k"], "--quant", e["quant"], "--dmg", e["dmg"])


def load_config(tag):
    return json.loads((MODELS / f"label_model_{tag}.json").read_text())


def predict(tag):
    """Apply the saved label model to the cached signals. Returns (pred idx, scores)."""
    import importlib
    lm = importlib.import_module("12_label_model")
    fin = importlib.import_module("15_final")
    cfg = load_config(tag)
    par = np.load(MODELS / f"label_model_{tag}.npz")
    sig, _ = lm.load_signals(list(cfg["signals"]))
    C, _ = lm.noise_covariates() if cfg.get("deconf") else (None, None)
    Z = 0
    for n, w in cfg["signals"].items():
        Zn = (sig[n] - par[f"mu_{n}"]) / par[f"sd_{n}"]
        if cfg.get("deconf"):
            Zn, _ = lm.residualize(Zn, C, coef=par[f"coef_{n}"])  # saved coefficients, no refit
        Z = Z + w * np.nan_to_num(Zn)
    if cfg["event"]:
        Z = Z + cfg["wevent"] * fin.event_signal(cfg["event"].split(","))
    X = lm.event_features()
    conf = lm.reliability() if cfg.get("conf") else np.ones(len(Z))
    F = lm.propagate(lm.knn_graph(X, cfg["k"]), Z, conf, cfg["alpha"], iters=200)
    F = F / F.std()
    S = F + par["bias"]
    return S.argmax(1), S


def write_submission(pred, path):
    ids = list_ids()
    df = pd.DataFrame({"Id": ids, "Phenomenon": [CLASSES[i] for i in pred]})
    assert len(df) == 3180 and df.Id.is_unique and df.Phenomenon.isin(CLASSES).all()
    df.to_csv(path, index=False)
    return df
