"""Compute the equal-weight wellness need index."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_Equity"))
from weighted_stats import weighted_quantiles

LIMITS = {
    "pm25": (15.4399995803833, 51.780006408691406),
    "tsi": (0.0923076942563057, 0.7461538910865784),
    "age60_share": (0.046576975476404825, 0.5223880597014925),
}


def need_index(indicators, population, reestimate_limits=False):
    normalized = {}
    for field, limits in LIMITS.items():
        values = np.asarray(indicators[field], dtype=float)
        lower, upper = weighted_quantiles(values, population, [0.01, 0.99]) if reestimate_limits else limits
        if upper <= lower:
            raise ValueError(f"{field} needs distinct percentile limits")
        normalized[field] = np.clip((values - lower) / (upper - lower), 0, 1)
    return np.mean(list(normalized.values()), axis=0), normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="Aligned grid NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ")
    parser.add_argument("--reestimate-limits", action="store_true")
    args = parser.parse_args()
    with np.load(args.grid, allow_pickle=False) as grid:
        population = grid["population"].astype(float)
        wni, indicators = need_index(grid, population, args.reestimate_limits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, wni=wni, **{f"x_{name}": values for name, values in indicators.items()})
    print(f"Population-weighted WNI: {np.average(wni, weights=population):.8f}")


if __name__ == "__main__":
    main()
