"""Compare the reported driving-time, supply-score, and aggregation scenarios."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "03_Accessibility"), str(CODE / "04_Equity")]
from equity import grid_provinces, high_need_income_gaps, read_counties
from indices import DIMENSIONS, relative_accessibility
from supply_demand import supply_crowding_gradient
from weighted_stats import (
    cluster_interval,
    confidence_index,
    county_mean,
    group_summaries,
    quantile_groups,
)


def scenarios(model, baseline):
    yield "baseline", 120, baseline
    with np.load(model / "accessibility_time_windows.npz", allow_pickle=False) as windows:
        for minutes in (60, 90):
            raw = {name: windows[f"A_{name}_{minutes}"] for name in DIMENSIONS}
            yield f"travel_{minutes}min", minutes, raw
    with np.load(model / "accessibility_sensitivity.npz", allow_pickle=False) as alternatives:
        yield "cold_season", 120, {
            "eco": alternatives["A_cold_eco"], "psy": alternatives["A_cold_psy"], "med": baseline["med"],
        }
        village = alternatives["A_village"]
        for score in (10, 50):
            yield f"village_score_{score}", 120, {**baseline, "psy": baseline["psy"] + (score / 30 - 1) * village}
        tcm = alternatives["A_tcm"]
        for score in (30, 60):
            yield f"tcm_score_{score}", 120, {**baseline, "med": baseline["med"] + (score / 50 - 1) * tcm}
    yield "arithmetic", 120, baseline


def coverage(county, zero_pop, replicates):
    population = county["population"].to_numpy(float)
    provinces = county["province_id"].to_numpy()
    high = county["income"].to_numpy() >= county["income"].median()
    age = pd.qcut(county["age60_share"], 3, labels=["low", "mid", "high"]).to_numpy()
    all_counties = np.arange(len(county))
    masks = {"national": np.full(len(county), True), "income_low": ~high, "income_high": high}
    masks.update({f"ageing_{level}": age == level for level in ("low", "mid", "high")})

    def rate(index, mask):
        idx = index[mask[index]]
        denom = population[idx].sum()
        return 100 * zero_pop[idx].sum() / denom if denom else np.nan

    for group, mask in masks.items():
        def statistic(index):
            return rate(index, mask)

        low, upper = cluster_interval(statistic, provinces, replicates)
        yield "zero_access_pct", group, statistic(all_counties), low, upper

    contrasts = (
        ("low_income_minus_high_income", masks["income_low"], masks["income_high"]),
        ("high_ageing_minus_low_ageing", masks["ageing_high"], masks["ageing_low"]),
    )
    for group, first, second in contrasts:
        def difference(index):
            return rate(index, first) - rate(index, second)

        low, upper = cluster_interval(difference, provinces, replicates)
        yield "zero_access_gap_pp", group, difference(all_counties), low, upper


def analyse(model, grid_file, county_file, output, replicates):
    county = read_counties(county_file)
    county_id = county["county_id"].to_numpy()
    county_pop = county["population"].to_numpy(float)
    county_province = county["province_id"].to_numpy()
    income = county["income"].to_numpy()
    with np.load(grid_file, allow_pickle=False) as grid:
        population = grid["population"].astype(float)
        grid_id = grid["county_id"]
        groups = quantile_groups(grid["wni"], population)
    if not np.array_equal(np.bincount(grid_id, weights=population)[county_id], county_pop):
        raise ValueError("Grid and county populations must agree")
    provinces = grid_provinces(grid_id, county)
    with np.load(model / "accessibility_main.npz", allow_pickle=False) as main:
        baseline = {name: main[f"A_{name}"] for name in DIMENSIONS}
        availability = {name: main[f"V_{name}"] for name in DIMENSIONS}

    output.mkdir(parents=True, exist_ok=True)
    supply_crowding_gradient(baseline, availability, population, groups, provinces, DIMENSIONS, replicates).to_csv(
        output / "supply_crowding_gradient.csv", index=False, float_format="%.12g",
    )
    rows = []
    for scenario, minutes, raw in scenarios(model, baseline):
        access = relative_accessibility(raw, population, arithmetic=scenario == "arithmetic")
        county_access = {d: county_mean(a, population, grid_id, county_id) for d, a in access.items()}

        def add(metric, resource, group, value, low, high):
            rows.append(dict(scenario=scenario, driving_minutes=minutes, metric=metric,
                             resource=resource, group=group, estimate=value, ci_low=low, ci_high=high))

        for name, values in access.items():
            means, boot = group_summaries(values, population, groups, provinces, replicates)
            low, high = np.quantile(boot[:, 5] - boot[:, 1], [0.025, 0.975])
            add("need_gradient", name, "national", means[5] - means[1], low, high)
        add("income_concentration", "C", "national",
            *confidence_index(county_access["C"], income, county_pop, county_province, replicates))
        if scenario in {"baseline", "travel_60min", "travel_90min"}:
            gaps = high_need_income_gaps(county, county_access, replicates)
            for row in gaps.itertuples(index=False):
                add("high_need_income_gap", row.measure, "low_income_minus_high_income", row.diff_HN, row.lo, row.hi)
            no_access = np.logical_or.reduce([raw[d] == 0 for d in DIMENSIONS])
            zero_pop = np.bincount(grid_id, weights=population * no_access)[county_id]
            for metric, group, value, low, high in coverage(county, zero_pop, replicates):
                add(metric, "any", group, value, low, high)
        print(f"Computed {scenario}", flush=True)
    pd.DataFrame(rows).to_csv(output / "Sensitivity_results.csv", index=False, float_format="%.12g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Directory of aligned accessibility NPZ results")
    parser.add_argument("--grid", type=Path, required=True, help="Aligned grid covariate NPZ")
    parser.add_argument("--county", type=Path, default=CODE.parent / "Data/County_equity.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    analyse(args.model, args.grid, args.county, args.output, args.replicates)


if __name__ == "__main__":
    main()
