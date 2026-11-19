"""National normalization and composite accessibility."""

import numpy as np

DIMENSIONS = ("eco", "psy", "med")
RESOURCES = (*DIMENSIONS, "C")
ACCESS_COLUMNS = dict(zip(RESOURCES, ("r_eco", "r_psy", "r_med", "C")))
EPSILON = 0.001


def relative_accessibility(raw, population, epsilon=EPSILON, arithmetic=False):
    relative = {}
    for name in DIMENSIONS:
        values = np.asarray(raw[name], dtype=float)
        reference = np.average(values, weights=population)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError(f"{name} has no positive national accessibility reference")
        relative[name] = values / reference
    stack = np.stack([relative[name] for name in DIMENSIONS])
    composite = stack.mean(axis=0) if arithmetic else np.exp(np.log(stack + epsilon).mean(axis=0))
    return {**relative, "C": composite}
