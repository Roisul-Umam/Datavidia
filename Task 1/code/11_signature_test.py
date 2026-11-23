"""Apakah sidik jari log rinci menambah sinyal di atas fitur saat ini?"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"E:\Kuliyah\Datavidia\Task 1\code")
from features import BASE
from features_sig import build_sig
from harness import evaluate, prepare

OUT = BASE / "exp" / "diagnostics"
H = prepare()
s_tr, s_te = build_sig()
assert (H.tr["shipment_id"] == s_tr["shipment_id"]).all()
cols = [c for c in s_tr.columns if c != "shipment_id" and c in s_te.columns]
Xs = s_tr.select(pl.col(cols).cast(pl.Float32)).to_numpy()
print("fitur sidik jari:", len(cols), flush=True)
res = {}
res["acuan"], _, _ = evaluate(H, "acuan")
for nm, pf in [("scan", "sg_"), ("rute", "sr_"), ("tagihan", "si_"), ("baris", "ss_")]:
    idx = [i for i, c in enumerate(cols) if c.startswith(pf)]
    res[nm], _, _ = evaluate(H, f"acuan+{nm} ({len(idx)})", Xs[:, idx], [cols[i] for i in idx])
res["semua"], _, imp = evaluate(H, f"acuan+semua ({len(cols)})", Xs, cols, top=30)
(OUT / "11_signature_test.json").write_text(json.dumps(res, indent=2))
