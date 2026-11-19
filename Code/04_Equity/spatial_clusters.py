"""Local co-location join counts with conditional randomization and BH adjustment."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from weighted_stats import benjamini_hochberg

TYPES = ("HH", "HL", "LH", "LL")
DRAWS = 10_000
ALPHA = 0.05


def neighbor_matrix(county):
    positions = {cid: i for i, cid in enumerate(county["county_id"])}
    rows, columns = [], []
    for i, field in enumerate(county["neighbor_ids"]):
        neighbors = [positions[int(token)] for token in field.split("|")]
        rows.extend([i] * len(neighbors))
        columns.extend(neighbors)
    n = len(county)
    graph = csr_matrix((np.ones(len(rows), dtype=int), (rows, columns)), shape=(n, n))
    if np.any(graph.diagonal()) or np.any(graph.data != 1):
        raise ValueError("Weights must be binary with no self-neighbors or duplicate edges")
    return graph


def distinct_neighbor_samples(labels, focal, degree, draws, rng):
    indices = rng.integers(0, len(labels) - 1, size=(draws, degree))
    if degree > 1:
        duplicate = np.flatnonzero(np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0, axis=1))
        while len(duplicate):
            proposal = rng.integers(0, len(labels) - 1, size=(len(duplicate), degree))
            indices[duplicate] = proposal
            duplicate = duplicate[np.any(np.diff(np.sort(proposal, axis=1), axis=1) == 0, axis=1)]
    indices += indices >= focal
    return labels[indices].sum(axis=1)


def simulate_tail(mask, members, local_counts, degree, ids, resource_index, type_index):
    tail = np.zeros(len(members), dtype=int)
    stream_seed = 42 + resource_index
    first_stream = np.random.default_rng(stream_seed)
    for j, focal in enumerate(members):
        neighbor_count = degree[focal]
        local = local_counts[focal]
        seed_args = [stream_seed, int(ids[focal]), DRAWS]
        if TYPES[type_index] == "HL":
            initial = np.asarray([
                first_stream.choice(len(mask) - 1, size=neighbor_count, replace=False)
                for _ in range(999)
            ])
            initial += initial >= focal
            first = np.count_nonzero(mask[initial].sum(axis=1) >= local)
            rng = np.random.default_rng(np.random.SeedSequence(seed_args))
            additional = distinct_neighbor_samples(mask, focal, neighbor_count, 9001, rng)
            tail[j] = first + np.count_nonzero(additional >= local)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([*seed_args, type_index + 1]))
            draw = distinct_neighbor_samples(mask, focal, neighbor_count, DRAWS, rng)
            tail[j] = np.count_nonzero(draw >= local)
    return tail


def analyse_clusters(county, access_columns, rerun=False):
    county = county.sort_values("county_id", kind="stable").reset_index(drop=True)
    graph = neighbor_matrix(county)
    degree = np.asarray(graph.sum(1)).ravel().astype(int)
    ids = county["county_id"].to_numpy()
    population = county["population"].to_numpy()
    high_need = county["wni"].to_numpy() > np.average(county["wni"], weights=population)
    results, summary = [], []
    for di, (resource, field) in enumerate(access_columns.items()):
        values = county[field].to_numpy()
        reference = np.average(values, weights=population) if resource == "C" else 1
        low_access = values < reference
        actual = np.select(
            [high_need & ~low_access, high_need & low_access, ~high_need & ~low_access],
            TYPES[:3], default="LL",
        )
        if not rerun and not np.array_equal(actual, county[f"match_{resource}"]):
            raise ValueError("County labels changed. Recompute the conditional randomizations.")
        for ti, kind in enumerate(TYPES):
            mask = actual == kind
            members = np.flatnonzero(mask)
            local_counts = np.asarray(graph @ mask.astype(int)).ravel()
            if rerun:
                tail = simulate_tail(mask, members, local_counts, degree, ids, di, ti)
            else:
                tail = county[f"upper_tail_{resource}"].to_numpy(dtype=int)[members]
            if np.any((tail < 0) | (tail > DRAWS)):
                raise ValueError("Upper-tail counts must be between zero and the number of draws")
            p_values = np.ones(len(county))
            p_values[members] = (tail + 1) / (DRAWS + 1)
            q_values = benjamini_hochberg(p_values)
            significant = mask & (q_values < ALPHA)
            results.append(pd.DataFrame({
                "county_id": ids[members], "resource": resource, "match_type": kind,
                "degree": degree[members], "same_type_neighbors": local_counts[members],
                "upper_tail_count": tail, "p_value": p_values[members], "q_value": q_values[members],
                "cluster_type": np.where(significant[members], kind, "NS"),
            }))
            residents = population[significant].sum()
            summary.append(dict(resource=resource, type=kind, counties=significant.sum(),
                                residents=residents, national_population_pct=100 * residents / population.sum()))
    local = pd.concat(results, ignore_index=True).sort_values(["resource", "county_id"])
    return local.reset_index(drop=True), pd.DataFrame(summary)
