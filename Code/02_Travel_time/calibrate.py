"""Calibrate friction-grid thresholds for the 60, 90 and 120 minute windows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = (60, 90, 120)
MAP_COLUMNS = ["amap_minutes", "baidu_minutes", "tencent_minutes"]


def threshold_scores(reference, friction, thresholds):
    order = np.argsort(friction, kind="stable")
    positive = reference[order].astype(int)
    count = np.searchsorted(friction[order], thresholds, side="right")
    tp = np.r_[0, np.cumsum(positive)][count].astype(float)
    fp = count - tp
    fn = positive.sum() - tp
    tn = len(positive) - positive.sum() - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn, denominator,
        out=np.full_like(tp, np.nan), where=denominator > 0,
    )
    return pd.DataFrame({
        "threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mcc": mcc, "weighted_loss": 2 * fp + fn,
    })


def segmented_fit(friction, map_time, breakpoint, limit=180):
    mask = (friction > 0) & (map_time > 0) & (friction < limit) & (map_time < limit)
    x, y = friction[mask], map_time[mask]
    if not len(x) or not x.min() < breakpoint < x.max() or np.ptp(y) == 0:
        raise ValueError("The fit requires varying paired times and data on both sides of the breakpoint")
    design = np.column_stack([np.ones_like(x), x, np.maximum(x - breakpoint, 0)])
    intercept, slope, change = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ (intercept, slope, change)
    r_squared = 1 - (residuals @ residuals) / np.sum((y - y.mean()) ** 2)
    return dict(pairs=len(x), maximum_minutes=limit, breakpoint=breakpoint,
                intercept=intercept, slope_below=slope, slope_above=slope + change, r_squared=r_squared)


def calibrate(pairs, output, replicates=500, seed=42):
    friction = pairs["friction_minutes"].to_numpy(float)
    map_time = pairs[MAP_COLUMNS].mean(axis=1).to_numpy(float)
    if not len(pairs) or not np.isfinite(friction).all() or not np.isfinite(map_time).all():
        raise ValueError("Every pair requires a finite friction time and a driving-time reference")
    if replicates < 1:
        raise ValueError("Use at least one bootstrap replicate")
    output.mkdir(parents=True, exist_ok=True)
    thresholds = np.arange(20, 141, dtype=float)
    results, samples = [], []
    for window in WINDOWS:
        reference = map_time <= window
        scores = threshold_scores(reference, friction, thresholds)
        scores.to_csv(output / f"calibration_search_{window}min.csv", index=False)
        selected = scores.loc[scores["weighted_loss"].idxmin()]
        rng = np.random.default_rng(seed)
        draws = np.empty(replicates)
        for i in range(replicates):
            index = rng.integers(0, len(pairs), len(pairs))
            scores_boot = threshold_scores(reference[index], friction[index], thresholds)
            draws[i] = scores_boot.loc[scores_boot["weighted_loss"].idxmin(), "threshold"]
        lower, upper = np.quantile(draws, [0.025, 0.975])
        results.append(dict(driving_minutes=window, **selected.to_dict(), ci_low=lower, ci_high=upper))
        samples.extend(dict(driving_minutes=window, replicate=i + 1, threshold=t) for i, t in enumerate(draws))
    table = pd.DataFrame(results)
    table.to_csv(output / "calibration_thresholds.csv", index=False)
    pd.DataFrame(samples).to_csv(output / "calibration_bootstrap.csv", index=False)
    fit = segmented_fit(friction, map_time, table.loc[table.driving_minutes.eq(120), "threshold"].item())
    pd.DataFrame([fit]).to_csv(output / "segmented_fit.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="CSV of paired friction and map minutes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    table = calibrate(pd.read_csv(args.pairs), args.output, args.replicates)
    print(table[["driving_minutes", "threshold", "mcc", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
