"""Need gradients, concentration indices, income matrices, and county clusters."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "03_Accessibility"))
from indices import ACCESS_COLUMNS, DIMENSIONS, RESOURCES
from spatial_clusters import analyse_clusters
from supply_demand import decompose_hl
from weighted_stats import (
    cluster_interval,
    confidence_index,
    group_summaries,
    quantile_groups,
    weighted_mean,
)


def read_counties(path):
    county = pd.read_csv(path).sort_values("county_id", kind="stable").reset_index(drop=True)
    if not county["county_id"].is_unique or not county["admin_code"].is_unique:
        raise ValueError("County identifiers must be unique")
    return county


def grid_provinces(county_id, county):
    mapped = pd.Index(county["county_id"]).get_indexer(county_id)
    if np.any(mapped == -1):
        raise ValueError("Each grid row must correspond to a released county")
    return county["province_id"].to_numpy()[mapped]


def grid_gradients(access, population, groups, provinces, replicates=1000, label="WNI"):
    records = []
    for name, values in access.items():
        mean, draws = group_summaries(values, population, groups, provinces, replicates)
        for q in range(1, 6):
            lower, upper = np.quantile(draws[:, q], [0.025, 0.975])
            records.append(dict(gradient=label, quintile=str(q), measure=name, mean=mean[q], lo=lower, hi=upper))
        lower, upper = np.quantile(draws[:, 5] - draws[:, 1], [0.025, 0.975])
        records.append(dict(gradient=label, quintile="Q5-Q1", measure=name,
                            mean=mean[5] - mean[1], lo=lower, hi=upper))
    return records


def county_equity(county, replicates=1000):
    pop = county["population"].to_numpy(float)
    provinces = county["province_id"].to_numpy()
    need = county["wni"].to_numpy()
    income = county["income"].to_numpy()
    need_q = quantile_groups(need, pop)
    income_q = quantile_groups(income, pop)
    need_ci, income_ci, deviations = [], [], []
    for resource, field in ACCESS_COLUMNS.items():
        access = county[field].to_numpy()
        estimate, low, high = confidence_index(access, need, pop, provinces, replicates)
        need_ci.append(dict(measure=resource, CI_need=estimate, lo=low, hi=high))
        for q in range(1, 6):
            layer = need_q == q
            estimate, low, high = confidence_index(
                access[layer], income[layer], pop[layer], provinces[layer], replicates
            )
            income_ci.append(dict(measure=resource, scope=f"WNI_Q{q}", n_county=layer.sum(),
                                  CI_income=estimate, lo=low, hi=high))
            reference = weighted_mean(access[layer], pop[layer])
            for k in range(1, 6):
                cell = layer & (income_q == k)
                mean = weighted_mean(access[cell], pop[cell]) if np.any(cell) else np.nan
                deviations.append(dict(
                    need_q=q, income_q=k, resource=resource, mean_access=mean,
                    need_layer_mean=reference, deviation_pct=100 * (mean / reference - 1),
                    counties=cell.sum(), residents=int(pop[cell].sum()),
                    population_pct_national=100 * pop[cell].sum() / pop.sum(),
                ))
    return (pd.DataFrame(need_ci), pd.DataFrame(income_ci), pd.DataFrame(deviations))


def high_need_income_gaps(county, access, replicates=1000):
    population = county["population"].to_numpy(float)
    province = county["province_id"].to_numpy()
    high_need = county["wni"].to_numpy() >= county["wni"].median()
    high_income = county["income"].to_numpy() >= county["income"].median()
    low = high_need & ~high_income
    high = high_need & high_income
    items = []
    for resource, values in access.items():
        def difference(index):
            a, b = low[index], high[index]
            if not a.any() or not b.any():
                return np.nan
            return (weighted_mean(values[index][a], population[index][a])
                    - weighted_mean(values[index][b], population[index][b]))

        lo, hi = cluster_interval(difference, province, replicates)
        items.append(dict(measure=resource, low_income_mean=weighted_mean(values[low], population[low]),
                          high_income_mean=weighted_mean(values[high], population[high]),
                          diff_HN=difference(np.arange(len(county))), lo=lo, hi=hi))
    return pd.DataFrame(items)


def analyse(data, output, replicates=1000, rerun=False, aligned_grid=None):
    county = read_counties(data / "County_equity.csv")
    grid = pd.read_csv(
        data / "Grid_accessibility.csv.gz",
        usecols=["county_id", "population", "wni", *ACCESS_COLUMNS.values()],
        dtype={"county_id": np.int32},
        float_precision="round_trip",
    )
    grid_id = grid["county_id"].to_numpy()
    population = grid["population"].to_numpy(float)
    provinces = grid_provinces(grid_id, county)
    access = {name: grid[field].to_numpy() for name, field in ACCESS_COLUMNS.items()}
    groups = quantile_groups(grid["wni"], population)
    gradients = grid_gradients(access, population, groups, provinces, replicates)
    if aligned_grid:
        ids = pd.read_csv(data / "Grid_accessibility.csv.gz", usecols=["row", "col"])
        with np.load(aligned_grid, allow_pickle=False) as source:
            if not all(np.array_equal(ids[axis], source[axis]) for axis in ("row", "col")):
                raise ValueError("The aligned grid must use the same cells and row order as the release")
            axes = {"PM2.5": "pm25", "TSI": "tsi", "Heat stress": "heat_stress",
                    "Cold stress": "cold_stress", "Ageing": "age60_share"}
            for label, field in axes.items():
                groups = quantile_groups(source[field], population)
                gradients += grid_gradients(access, population, groups, provinces, replicates, label)

    need, income, matrix = county_equity(county, replicates)
    county_access = {d: county[ACCESS_COLUMNS[d]].to_numpy() for d in RESOURCES}
    gaps = high_need_income_gaps(county, county_access, replicates)
    local, clusters = analyse_clusters(county, ACCESS_COLUMNS, rerun)
    details, structure = decompose_hl(county, local, DIMENSIONS)
    tables = {
        "need_gradients": pd.DataFrame(gradients),
        "need_concentration": need,
        "income_concentration": income,
        "income_accessibility_deviations": matrix,
        "high_need_income_gaps": gaps,
        "join_count_tests": local,
        "spatial_cluster_summary": clusters,
        "hl_county_supply_demand": details,
        "hl_supply_demand_summary": structure,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    print(f"Computed {len(tables)} tables for {len(grid):,} grid cells and {len(county):,} counties")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CODE.parent / "Data")
    parser.add_argument("--output", type=Path, default=Path("outputs/equity"))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--rerun-permutations", action="store_true")
    parser.add_argument("--aligned-grid", type=Path, help="Optional covariate NPZ for component gradients")
    args = parser.parse_args()
    analyse(args.data, args.output, args.replicates, args.rerun_permutations, args.aligned_grid)


if __name__ == "__main__":
    main()
