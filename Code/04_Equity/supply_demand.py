"""Supply and population-competition diagnosis for significant H-L counties."""

import numpy as np
import pandas as pd

from weighted_stats import group_summaries, weighted_mean


def decompose_hl(county, clusters, dimensions):
    selected_cases, summaries = [], []
    for resource in dimensions:
        hl = clusters["resource"].eq(resource) & clusters["cluster_type"].eq("HL")
        focus = county[county["county_id"].isin(clusters.loc[hl, "county_id"])]
        vbar = weighted_mean(county[f"V_{resource}"], county["population"])
        v_rel = focus[f"V_{resource}"].to_numpy() / vbar
        relative = focus[f"r_{resource}"].to_numpy()
        positive = relative > 0
        scarcity = np.full(len(focus), np.nan)
        competition = np.full(len(focus), np.nan)
        scarcity[positive] = -np.log(v_rel[positive])
        competition[positive] = np.log(v_rel[positive] / relative[positive])
        structure = np.full(len(focus), "zero", dtype=object)
        structure[positive] = np.where(scarcity[positive] > competition[positive], "scarcity", "competition")
        count = focus["population"].sum()
        selected_cases.append(pd.DataFrame({
            "county_id": focus["county_id"].to_numpy(), "resource": resource,
            "population": focus["population"].to_numpy(), "relative_accessibility": relative,
            "availability": focus[f"V_{resource}"].to_numpy(),
            "shared_population": focus[f"K_{resource}"].to_numpy(),
            "scarcity_term": scarcity, "competition_term": competition, "structure": structure,
        }))
        for kind in ("scarcity", "competition", "zero"):
            mask = structure == kind
            residents = focus.loc[mask, "population"].sum()
            share = 100 * residents / count if count else np.nan
            summaries.append(dict(resource=resource, structure=kind, counties=mask.sum(),
                                  residents=residents, share_HL_residents_pct=share,
                                  HL_counties=len(focus), HL_residents=count))
    return pd.concat(selected_cases, ignore_index=True), pd.DataFrame(summaries)


def supply_crowding_gradient(raw, availability, population, groups, provinces, dimensions, replicates=1000):
    rows = []
    for resource in dimensions:
        a, a_boot = group_summaries(raw[resource], population, groups, provinces, replicates)
        v, v_boot = group_summaries(availability[resource], population, groups, provinces, replicates)
        a_ref = weighted_mean(raw[resource], population)
        v_ref = weighted_mean(availability[resource], population)
        for q in range(1, 6):
            v_low, v_high = np.nanquantile(v_boot[:, q] / v_ref, [0.025, 0.975])
            a_sample = a_boot[:, q]
            ratios = np.divide(v_boot[:, q], a_sample, out=np.full_like(a_sample, np.nan), where=a_sample > 0)
            k_low, k_high = np.nanquantile(ratios / (v_ref / a_ref), [0.025, 0.975])
            rows.append(dict(measure=resource, quintile=q, V_rel=v[q] / v_ref, V_lo=v_low, V_hi=v_high,
                             K_rel=(v[q] / a[q]) / (v_ref / a_ref) if a[q] else np.nan,
                             K_lo=k_low, K_hi=k_high, r=a[q] / a_ref))
    return pd.DataFrame(rows)
