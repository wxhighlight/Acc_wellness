"""Population-based Gaussian two-step floating catchment area model."""

import argparse
import math
from pathlib import Path
import sys

import numba as nb
from numba import njit, prange
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_Travel_time"))
from travel_time import dijkstra_one, neighbor_dists

@njit(parallel=True, cache=True)
def phase_a(fr, H, W, dx, dy, dd, src_r, src_c, start_t, tmax, idx_map, D, Ts, dcol, beta, dist, hk, hv, touched, buf_idx, buf_t, cap, n_out, overflow, denom, ntouch_out):
    B = src_r.shape[0]
    nthr = dist.shape[0]
    NS = Ts.shape[0]
    for t in prange(nthr):
        for i in range(t, B, nthr):
            n, ntch, ov = dijkstra_one(fr, H, W, dx, dy, dd, src_r[i], src_c[i], tmax, dist[t], hk[t], hv[t], touched[t], idx_map, buf_idx[i], buf_t[i], cap)
            n_out[i] = n
            overflow[i] = ov
            ntouch_out[i] = ntch
            if ov:
                continue
            s_idx = idx_map[src_r[i] * W + src_c[i]]
            for e in range(n):
                if buf_idx[i, e] == s_idx:
                    buf_t[i, e] = start_t[i]
            for sidx in range(NS):
                T = Ts[sidx]
                col = dcol[sidx]
                acc = 0.0
                for e in range(n):
                    tt = buf_t[i, e]
                    if tt > 0.0 and tt <= T:
                        x = tt / T
                        acc += D[buf_idx[i, e], col] * math.exp(-beta * x * x)
                denom[i, sidx] = acc

@njit(parallel=True, cache=True)
def phase_b(buf_idx, buf_t, n_out, Wt, layer_scen, Ts, beta, A):
    NL = A.shape[0]
    B = buf_idx.shape[0]
    for l in prange(NL):
        T = Ts[layer_scen[l]]
        for i in range(B):
            w = Wt[i, l]
            if w == 0.0:
                continue
            for e in range(n_out[i]):
                tt = buf_t[i, e]
                if tt > 0.0 and tt <= T:
                    x = tt / T
                    A[l, buf_idx[i, e]] += w * math.exp(-beta * x * x)

class Engine:

    def __init__(self, fr, transform, idx_map, D, scen, comps, beta=1.5, nthreads=None, batch=512, cap=400000, heap_cap=6000000, log=print):
        self.H, self.W = fr.shape
        self.fr = np.ascontiguousarray(fr.ravel().astype(np.float32))
        self.dx, self.dy, self.dd = neighbor_dists(transform, self.H)
        self.idx_map = np.ascontiguousarray(idx_map.ravel().astype(np.int32))
        self.D = np.ascontiguousarray(D, dtype=np.float64)
        self.scen = scen
        self.Ts = np.array([s[1] for s in scen], np.float64)
        self.dcol = np.array([s[2] for s in scen], np.int64)
        self.dmins = np.array([s[3] if len(s) > 3 else 0.0 for s in scen], np.float64)
        self.modes = [s[4] if len(s) > 4 else 'ratio' for s in scen]
        self.comps = comps
        self.NS, self.NC = (len(scen), len(comps))
        self.layer_scen = np.repeat(np.arange(self.NS), self.NC).astype(np.int64)
        self.beta = float(beta)
        self.nthr = nthreads or nb.get_num_threads()
        self.batch, self.cap, self.heap_cap = (batch, cap, heap_cap)
        self.log = log
        self.tmax = np.float32(self.Ts.max())
        n_cells = int(self.idx_map.max()) + 1
        self.A = np.zeros((self.NS * self.NC, n_cells), np.float64)

    def start_time(self, r, c):
        f = self.fr[r * self.W + c]
        return max(float(f) * (float(self.dx[r]) + float(self.dy[r])) / 2.0 * 0.5, 0.001)

    def run(self, src_r, src_c, S, F=None):
        F = F or {}
        import time
        n = len(src_r)
        nthr = self.nthr
        HW = self.H * self.W
        dist = np.full((nthr, HW), np.inf, np.float32)
        hk = np.empty((nthr, self.heap_cap), np.float32)
        hv = np.empty((nthr, self.heap_cap), np.int64)
        touched = np.empty((nthr, min(HW, 12000000)), np.int64)
        denom_all = np.zeros((n, self.NS))
        nreach_all = np.zeros(n, np.int64)
        ntouch_all = np.zeros(n, np.int64)
        st = np.array([self.start_time(r, c) for r, c in zip(src_r, src_c)], np.float32)
        order = np.random.default_rng(0).permutation(n)
        t0 = time.time()
        B = self.batch
        buf_idx = np.empty((B, self.cap), np.int32)
        buf_t = np.empty((B, self.cap), np.float32)
        done = 0
        for b0 in range(0, n, B):
            sel = order[b0:b0 + B]
            nb_ = len(sel)
            n_out = np.zeros(nb_, np.int64)
            ov = np.zeros(nb_, np.bool_)
            den = np.zeros((nb_, self.NS))
            ntch = np.zeros(nb_, np.int64)
            sr, sc = (src_r[sel].astype(np.int64), src_c[sel].astype(np.int64))
            phase_a(self.fr, self.H, self.W, self.dx, self.dy, self.dd, sr, sc, st[sel], self.tmax, self.idx_map, self.D, self.Ts, self.dcol, self.beta, dist, hk, hv, touched, buf_idx[:nb_], buf_t[:nb_], self.cap, n_out, ov, den, ntch)
            if ov.any():
                raise RuntimeError('Search buffer capacity exceeded; increase cap, heap_cap, or touched capacity.')
            Wt = np.zeros((nb_, self.NS * self.NC))
            Ssel = S[sel]
            for s in range(self.NS):
                Ss = Ssel * F[s][sel] if s in F else Ssel
                blk = slice(s * self.NC, (s + 1) * self.NC)
                if self.modes[s] == 'avail':
                    Wt[:, blk] = Ss
                    continue
                d = den[:, s]
                dm = self.dmins[s]
                ok = d > 1e-12 if dm <= 0 else np.ones(nb_, bool)
                Wt[ok, blk] = Ss[ok] / np.maximum(d[ok], dm)[:, None]
            phase_b(buf_idx[:nb_], buf_t[:nb_], n_out, Wt, self.layer_scen, self.Ts, self.beta, self.A)
            denom_all[sel] = den
            nreach_all[sel] = n_out
            ntouch_all[sel] = ntch
            done += nb_
            el = time.time() - t0
            self.log(f'{done:,}/{n:,} source cells; {el / 60:.1f} min; ETA {el / done * (n - done) / 60:.1f} min')
        return (denom_all, nreach_all, ntouch_all)




COMPONENTS = {
    "eco": ("eco_alevel", "eco_forest", "eco_wetland"),
    "psy": ("psy_alevel", "psy_village_trad", "psy_village_tour", "psy_village_hist"),
    "med": ("med_hospital", "med_nursing", "med_tcm"),
}
THRESHOLDS = {60: 41, 90: 70, 120: 100}
BETA = 1.5
DEMAND_FLOOR = 1000.0


def run_model(inputs, output, threads=4, batch_size=256):
    with np.load(inputs / "grid_covariates.npz", allow_pickle=False) as grid:
        row = grid["row"]
        col = grid["col"]
        population = grid["population"].astype(float)
    with rasterio.open(inputs / "friction_minutes_per_metre.tif") as raster:
        if raster.crs is None or raster.crs.to_epsg() != 4326:
            raise ValueError("Use the aligned EPSG:4326 friction grid")
        friction = raster.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = raster.transform
    friction[(friction <= 0) | ~np.isfinite(friction)] = np.nan
    index = np.full(friction.shape, -1, dtype=np.int32)
    index[row, col] = np.arange(len(row), dtype=np.int32)
    sources = pd.read_csv(inputs / "resource_sources.csv")
    components = [c[2:] for c in sources if c.startswith("S_")]
    included = {c for dimension in COMPONENTS.values() for c in dimension}
    components = [c for c in components if c in included]
    if set(components) != included:
        raise ValueError("Supply scores are required for all ten modeled components")
    supply = sources[[f"S_{c}" for c in components]].to_numpy(float)
    source_row = sources["row"].to_numpy(dtype=int)
    source_col = sources["col"].to_numpy(dtype=int)
    if not np.isfinite(friction[source_row, source_col]).all():
        raise ValueError("All source cells must be on the traversable friction grid")
    cold_share = sources["cold_stress"].to_numpy(float)
    if not np.all((cold_share >= 0) & (cold_share <= 1)):
        raise ValueError("Source cold-stress shares must be between zero and one")
    cold = np.ones_like(supply)
    outdoor = [i for i, c in enumerate(components) if c.startswith(("eco_", "psy_"))]
    cold[:, outdoor] = 1 - cold_share[:, None]

    scenarios = [
        ("baseline", THRESHOLDS[120], 0, DEMAND_FLOOR),
        ("travel_60min", THRESHOLDS[60], 0, DEMAND_FLOOR),
        ("travel_90min", THRESHOLDS[90], 0, DEMAND_FLOOR),
        ("availability", THRESHOLDS[120], 0, 0, "avail"),
        ("cold_season", THRESHOLDS[120], 0, DEMAND_FLOOR),
    ]
    nb.set_num_threads(threads)
    engine = Engine(
        friction, transform, index, population[:, None], scenarios, components,
        beta=BETA, nthreads=threads, batch=batch_size,
    )
    engine.run(source_row, source_col, supply, F={4: cold})
    layers = engine.A.astype(np.float32)
    del engine

    def component(case, name):
        return layers[case * len(components) + components.index(name)].astype(float)

    def total(case, resource):
        return sum(component(case, name) for name in COMPONENTS[resource])

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "accessibility_main.npz",
        **{f"A_{d}": total(0, d) for d in COMPONENTS},
        **{f"V_{d}": total(3, d) for d in COMPONENTS},
    )
    np.savez_compressed(
        output / "accessibility_time_windows.npz",
        **{f"A_{d}_{t}": total(i, d) for i, t in ((1, 60), (2, 90)) for d in COMPONENTS},
    )
    np.savez_compressed(
        output / "accessibility_sensitivity.npz",
        A_village=sum(component(0, c) for c in COMPONENTS["psy"][1:]),
        A_tcm=component(0, "med_tcm"),
        A_cold_eco=total(4, "eco"),
        A_cold_psy=total(4, "psy"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Directory of aligned modeling inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run_model(args.inputs, args.output, args.threads, args.batch_size)


if __name__ == "__main__":
    main()


