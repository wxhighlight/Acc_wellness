"""Population-weighted summaries and province cluster resampling."""
import numpy as np


def weighted_mean(values, weights):
    return float(np.average(np.asarray(values, float), weights=np.asarray(weights, float)))


def weighted_quantiles(values, weights, probabilities):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return np.interp(probabilities, cumulative, values[order])


def quantile_groups(values, weights, groups=5):
    edges = weighted_quantiles(values, weights, np.arange(1, groups) / groups)
    return np.searchsorted(edges, values, side="right") + 1


def fractional_rank(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), float)
    rank[order] = (np.cumsum(weights[order]) - weights[order] / 2) / weights.sum()
    return rank


def concentration_index(values, ranking, weights):
    values, ranking, weights = map(lambda x: np.asarray(x, float), (values, ranking, weights))
    mask = np.isfinite(values) & np.isfinite(ranking) & np.isfinite(weights) & (weights > 0)
    values, ranking, weights = values[mask], ranking[mask], weights[mask]
    if len(values) < 3:
        return np.nan
    mean = weighted_mean(values, weights)
    if mean == 0:
        return np.nan
    rank = fractional_rank(ranking, weights)
    covariance = np.average((values - mean) * (rank - weighted_mean(rank, weights)), weights=weights)
    return float(2 * covariance / mean)


def province_multiplicity(provinces, replicates=1000, seed=42):
    levels = np.unique(provinces)
    rng = np.random.default_rng(seed)
    draws = np.array([np.bincount(rng.integers(0, len(levels), len(levels)), minlength=len(levels))
                      for _ in range(replicates)], dtype=float)
    return levels, draws


def group_summaries(values, population, groups, provinces, replicates=1000, seed=42, n_groups=5):
    """Keep original group boundaries and national reference values fixed."""
    values, population = np.asarray(values, float), np.asarray(population, float)
    levels, draws = province_multiplicity(provinces, replicates, seed)
    pi = np.searchsorted(levels, provinces)
    keys = pi * (n_groups + 1) + groups
    shape = (len(levels), n_groups + 1)
    denominator = np.bincount(keys, weights=population, minlength=np.prod(shape)).reshape(shape)
    numerator = np.bincount(keys, weights=population * values, minlength=np.prod(shape)).reshape(shape)
    estimate = numerator.sum(0) / np.maximum(denominator.sum(0), 1e-300)
    boot = (draws @ numerator) / np.maximum(draws @ denominator, 1e-300)
    return estimate, boot


def cluster_interval(statistic, provinces, replicates=1000, seed=42):
    """Resample represented provinces and retain all their county observations."""
    provinces = np.asarray(provinces)
    levels = np.unique(provinces)
    members = {p: np.flatnonzero(provinces == p) for p in levels}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    for b in range(replicates):
        selected = rng.choice(levels, len(levels), replace=True)
        index = np.concatenate([members[p] for p in selected])
        values[b] = statistic(index)
    return np.nanquantile(values, [.025, .975])


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values, kind="stable")
    adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def confidence_index(values, ranking, population, provinces, replicates=1000):
    estimate = concentration_index(values, ranking, population)

    def statistic(index):
        return concentration_index(values[index], ranking[index], population[index])

    lower, upper = cluster_interval(statistic, provinces, replicates)
    return estimate, lower, upper


def county_mean(values, population, county_id, selected_ids):
    size = int(np.max(county_id)) + 1
    total = np.bincount(county_id, weights=population, minlength=size)
    numerator = np.bincount(county_id, weights=population * values, minlength=size)
    return numerator[selected_ids] / total[selected_ids]

