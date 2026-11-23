"""Forecast PM2.5 AQI harian Jakarta, 1 Januari - 31 Maret 2026.

Model: kurva musiman tahunan (regresi median) + koreksi hari libur.
Jalankan dari folder warmup: python solution.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HOLIDAY_GROUPS = ["Idul Fitri", "Imlek", "Nyepi"]
HOLIDAY_SHRINK = 10
ORIGIN_GAP_DAYS = 3  # train berakhir dengan 3 hari target kosong, backtest menirunya


def holiday_group(names: pd.Series) -> pd.Series:
    names = names.fillna("")
    groups = pd.Series("", index=names.index)
    for group in HOLIDAY_GROUPS:
        groups[names.str.contains(group)] = group
    return groups


def design_matrix(dates: pd.DatetimeIndex) -> np.ndarray:
    angle = 2 * np.pi * dates.dayofyear.values / 365.25
    return np.column_stack([np.ones(len(angle)), np.sin(angle), np.cos(angle)])


def fit_median_curve(history: pd.Series, iterations: int = 60) -> np.ndarray:
    """Regresi L1 lewat IRLS, karena MAE paling kecil saat memprediksi median."""
    history = history.dropna()
    X = design_matrix(history.index)
    y = history.values
    weights = np.ones(len(y))
    ridge = 1e-3 * np.eye(X.shape[1])
    for _ in range(iterations):
        beta = np.linalg.solve((X.T * weights) @ X + ridge, (X.T * weights) @ y)
        weights = 1 / np.maximum(np.abs(y - X @ beta), 1.0)
    return beta


def holiday_effect(history, history_groups, future_groups) -> np.ndarray:
    """Efek libur dibanding median 29 hari sekitarnya, dikecilkan bila contohnya sedikit."""
    local_median = history.rolling(29, center=True, min_periods=12).median()
    residual = (history - local_median).dropna()
    groups = history_groups.reindex(residual.index)
    effect = np.zeros(len(future_groups))
    for group in HOLIDAY_GROUPS:
        values = residual[groups == group]
        if len(values):
            shrink = len(values) / (len(values) + HOLIDAY_SHRINK)
            effect[future_groups.values == group] = values.median() * shrink
    return effect


def forecast(history, history_groups, future_dates, future_groups) -> np.ndarray:
    beta = fit_median_curve(history)
    seasonal = design_matrix(future_dates) @ beta
    return seasonal + holiday_effect(history, history_groups, future_groups)


def backtest(y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for year in (2023, 2024, 2025):
        origin = pd.Timestamp(f"{year}-01-01")
        dates = pd.date_range(origin, f"{year}-03-31")
        history = y[y.index < origin - pd.Timedelta(days=ORIGIN_GAP_DAYS)]
        actual = y.reindex(dates)
        scored = actual.notna().values
        pred = forecast(history, groups, dates, groups.reindex(dates).fillna(""))
        naive = np.full(len(dates), history.dropna().iloc[-28:].median())
        rows.append(
            {
                "periode": f"Jan-Mar {year}",
                "MAE_model": np.abs(actual.values - pred)[scored].mean(),
                "MAE_median_28_hari": np.abs(actual.values - naive)[scored].mean(),
                "MAE_konstanta_104": np.abs(actual.values - 104)[scored].mean(),
                "bias_model": (pred - actual.values)[scored].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["tanggal"]).set_index("tanggal")
    test = pd.read_csv(DATA / "test.csv", parse_dates=["tanggal"])
    sample = pd.read_csv(DATA / "sample_submission.csv")

    y = train["pm25"]
    groups = holiday_group(train["nama_libur"])
    result = backtest(y, groups)
    print(result.round(2).to_string(index=False))
    print("Rata-rata:", result.drop(columns="periode").mean().round(2).to_dict())

    future_dates = pd.DatetimeIndex(test["tanggal"])
    future_groups = holiday_group(test["nama_libur"])
    pred = forecast(y, groups, future_dates, future_groups)

    submission = pd.DataFrame({"id": test["id"], "pm25": np.clip(np.rint(pred), 0, 500).astype(int)})
    submission = sample[["id"]].merge(submission, on="id", how="left", validate="one_to_one")
    submission.to_csv(ROOT / "submission.csv", index=False)
    print("Prediksi:", submission["pm25"].describe().round(1).to_dict())
    print("Rata-rata per bulan:", submission.groupby(submission["id"].str[:7])["pm25"].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
