"""Forecast PM2.5 AQI harian Jakarta versi 2, 1 Januari - 31 Maret 2026.

Prediksi adalah rata-rata 50:50 dari dua skenario:
1. Level kembali normal: kurva musiman + efek libur (model versi 1).
2. Level rendah akhir 2025 berlanjut: rata-rata 903 konfigurasi kurva
   dengan koreksi level terbaru. Semua konfigurasi dipakai tanpa memilih
   yang terbaik, supaya tidak overfit pada periode validasi yang sedikit.

Jalankan dari folder warmup: python solution_v2.py
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "submission" / "submission_2.csv"

HOLIDAY_GROUPS = ["Idul Fitri", "Imlek", "Nyepi"]
HOLIDAY_SHRINK = 10
ORIGIN_GAP_DAYS = 3  # train berakhir dengan 3 hari target kosong, backtest menirunya
BLEND_WEIGHT = 0.5

# (jumlah harmonik, half-life bobot dalam hari, fit pada skala log)
CURVES = [
    (1, None, False),
    (1, 730, False),
    (1, 365, False),
    (1, None, True),
    (1, 730, True),
    (2, None, False),
    (2, 730, False),
]
LOOKBACKS = (28, 56, 90, 120)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
DECAYS = (30, 60, 120, np.inf)
MODES = ("add", "mult")


def holiday_group(names: pd.Series) -> pd.Series:
    names = names.fillna("")
    groups = pd.Series("", index=names.index)
    for group in HOLIDAY_GROUPS:
        groups[names.str.contains(group)] = group
    return groups


def design_matrix(dates: pd.DatetimeIndex, harmonics: int) -> np.ndarray:
    angle = 2 * np.pi * dates.dayofyear.values / 365.25
    columns = [np.ones(len(angle))]
    for order in range(1, harmonics + 1):
        columns += [np.sin(order * angle), np.cos(order * angle)]
    return np.column_stack(columns)


def predict_curve(history, dates, harmonics=1, half_life=None, log_scale=False) -> np.ndarray:
    """Regresi median (L1) lewat IRLS, karena MAE paling kecil saat memprediksi median."""
    history = history.dropna()
    X = design_matrix(history.index, harmonics)
    target = np.log1p(history.values) if log_scale else history.values
    prior = np.ones(len(target))
    if half_life:
        prior = 0.5 ** ((history.index.max() - history.index).days.values / half_life)
    weights = prior.copy()
    ridge = 1e-3 * np.eye(X.shape[1])
    for _ in range(60):
        beta = np.linalg.solve((X.T * weights) @ X + ridge, (X.T * weights) @ target)
        weights = prior / np.maximum(np.abs(target - X @ beta), 1.0)
    prediction = design_matrix(dates, harmonics) @ beta
    return np.expm1(prediction) if log_scale else prediction


def holiday_effect(history, groups, dates) -> np.ndarray:
    """Efek libur dibanding median 29 hari sekitarnya, dikecilkan bila contohnya sedikit."""
    residual = (history - history.rolling(29, center=True, min_periods=12).median()).dropna()
    past_groups = groups.reindex(residual.index)
    future_groups = groups.reindex(dates).fillna("").values
    effect = np.zeros(len(dates))
    for group in HOLIDAY_GROUPS:
        values = residual[past_groups == group]
        if len(values):
            effect[future_groups == group] = values.median() * len(values) / (len(values) + HOLIDAY_SHRINK)
    return effect


def recent_level_gap(observed, curve, lookback):
    """Selisih level terbaru terhadap kurva yang hanya dilatih dari data lebih lama."""
    recent = observed.iloc[-lookback:]
    older = observed[observed.index < recent.index.min()]
    if len(older) < 120:
        return 0.0, 0.0
    expected = np.maximum(predict_curve(older, recent.index, *curve), 1.0)
    additive = float(np.median(recent.values - expected))
    multiplicative = float(np.median(np.log((recent.values + 1) / (expected + 1))))
    return additive, multiplicative


def forecast_scenarios(history, groups, dates):
    """Mengembalikan prediksi skenario normal dan skenario level berlanjut."""
    observed = history.dropna()
    horizon = (dates - observed.index.max()).days.values.astype(float)
    holiday = holiday_effect(history, groups, dates)
    members = []
    for curve in CURVES:
        base = predict_curve(history, dates, *curve)
        members.append(base + holiday)
        gaps = {lookback: recent_level_gap(observed, curve, lookback) for lookback in LOOKBACKS}
        for lookback, weight, decay, mode in product(LOOKBACKS, WEIGHTS, DECAYS, MODES):
            additive, multiplicative = gaps[lookback]
            fade = weight * np.exp(-horizon / decay)
            adjusted = base + additive * fade if mode == "add" else base * np.exp(multiplicative * fade)
            members.append(adjusted + holiday)
    members = np.clip(np.array(members), 0, 500)
    return members[0], members.mean(axis=0)


def backtest(y, groups) -> pd.DataFrame:
    jfm_folds = {year: pd.date_range(f"{year}-01-01", f"{year}-03-31") for year in (2023, 2024, 2025)}
    rolling_folds = [pd.date_range(start, periods=90) for start in pd.date_range("2023-01-01", "2025-10-01", freq="MS")]

    def fold_scores(dates):
        history = y[y.index < dates[0] - pd.Timedelta(days=ORIGIN_GAP_DAYS)]
        actual = y.reindex(dates).values
        scored = ~np.isnan(actual)
        normal, persistent = forecast_scenarios(history, groups, dates)
        blended = (1 - BLEND_WEIGHT) * normal + BLEND_WEIGHT * persistent
        return [np.abs(actual - pred)[scored].mean() for pred in (normal, persistent, blended)]

    jfm = np.array([fold_scores(dates) for dates in jfm_folds.values()])
    rolling = np.array([fold_scores(dates) for dates in rolling_folds])
    table = pd.DataFrame(jfm.T, columns=[f"Jan-Mar {year}" for year in jfm_folds], index=["v1 normal", "level berlanjut", "v2 blend 50:50"])
    table["rata2 Jan-Mar"] = jfm.mean(axis=0)
    table["terburuk"] = jfm.max(axis=0)
    table["rolling 34"] = rolling.mean(axis=0)
    table["menang vs v1"] = (rolling < rolling[:, [0]]).sum(axis=0)
    return table


def main() -> None:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["tanggal"]).set_index("tanggal").sort_index()
    test = pd.read_csv(DATA / "test.csv", parse_dates=["tanggal"])
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["pm25"]
    dates = pd.DatetimeIndex(test["tanggal"])
    groups = pd.concat([holiday_group(train["nama_libur"]), pd.Series(holiday_group(test["nama_libur"]).values, index=dates)])

    print(backtest(y, groups).round(2).to_string())

    normal, persistent = forecast_scenarios(y, groups, dates)
    blended = (1 - BLEND_WEIGHT) * normal + BLEND_WEIGHT * persistent
    submission = pd.DataFrame({"id": test["id"], "pm25": np.clip(np.rint(blended), 0, 500).astype(int)})
    submission = sample[["id"]].merge(submission, on="id", how="left", validate="one_to_one")
    OUTPUT.parent.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT, index=False)

    print(f"\nRata-rata prediksi: normal {normal.mean():.1f}, level berlanjut {persistent.mean():.1f}, v2 {blended.mean():.1f}")
    print("v2 per bulan:", submission.groupby(submission["id"].str[:7])["pm25"].mean().round(1).to_dict())
    print(f"Rentang v2: {submission['pm25'].min()}-{submission['pm25'].max()}")


if __name__ == "__main__":
    main()
