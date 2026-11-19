"""Eight-neighbor shortest paths on the travel-time friction grid."""

import math

import numpy as np
from numba import njit

def neighbor_dists(transform, H):
    xres, yres = (abs(transform.a), abs(transform.e))
    dx = np.zeros(H, np.float32)
    dy = np.zeros(H, np.float32)
    dd = np.zeros(H, np.float32)
    for r in range(H):
        lat = transform.f + (r + 0.5) * transform.e
        la = np.deg2rad(lat)
        m_lat = 111132.92 - 559.82 * np.cos(2 * la) + 1.175 * np.cos(4 * la) - 0.0023 * np.cos(6 * la)
        m_lon = 111412.84 * np.cos(la) - 93.5 * np.cos(3 * la) + 0.118 * np.cos(5 * la)
        dx[r] = xres * m_lon
        dy[r] = yres * m_lat
        dd[r] = math.sqrt(float(dx[r]) ** 2 + float(dy[r]) ** 2)
    return (dx, dy, dd)

@njit(cache=True, inline='always')
def _push(hk, hv, n, k, v):
    i = n
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = i - 1 >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = (hk[i], hk[p])
        hv[p], hv[i] = (hv[i], hv[p])
        i = p
    return n + 1

@njit(cache=True, inline='always')
def _pop(hk, hv, n):
    k, v = (hk[0], hv[0])
    n -= 1
    hk[0] = hk[n]
    hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        r = l + 1
        m = l
        if r < n and hk[r] < hk[l]:
            m = r
        if hk[i] <= hk[m]:
            break
        hk[m], hk[i] = (hk[i], hk[m])
        hv[m], hv[i] = (hv[i], hv[m])
        i = m
    return (k, v, n)

@njit(cache=True)
def dijkstra_one(fr, H, W, dx, dy, dd, sr, sc, tmax, dist, hk, hv, touched, idx_map, out_idx, out_t, cap):
    DR = (-1, -1, -1, 0, 0, 1, 1, 1)
    DC = (-1, 0, 1, -1, 1, -1, 0, 1)
    s = sr * W + sc
    dist[s] = 0.0
    ntouch = 0
    touched[ntouch] = s
    ntouch += 1
    nh = _push(hk, hv, 0, np.float32(0.0), s)
    hcap = hk.shape[0]
    nout = 0
    over = False
    while nh > 0 and not over:
        d, u, nh = _pop(hk, hv, nh)
        if d > dist[u]:
            continue
        r = u // W
        c = u - r * W
        f1 = fr[u]
        k = idx_map[u]
        if k >= 0:
            if nout < cap:
                out_idx[nout] = k
                out_t[nout] = d
            else:
                over = True
            nout += 1
        for q in range(8):
            nr = r + DR[q]
            nc = c + DC[q]
            if nr < 0 or nr >= H or nc < 0 or (nc >= W):
                continue
            v = nr * W + nc
            f2 = fr[v]
            if not f2 == f2:
                continue
            if DR[q] != 0 and DC[q] != 0:
                step = dd[r]
            elif DR[q] == 0:
                step = dx[r]
            else:
                step = dy[r]
            nt = np.float32(d + np.float32((f1 + f2) * np.float32(0.5)) * step)
            if nt > tmax:
                continue
            if nt < dist[v]:
                if dist[v] == np.inf:
                    if ntouch >= touched.shape[0]:
                        over = True
                        break
                    touched[ntouch] = v
                    ntouch += 1
                dist[v] = nt
                if nh < hcap:
                    nh = _push(hk, hv, nh, nt, v)
                else:
                    over = True
                    break
    for i in range(ntouch):
        dist[touched[i]] = np.inf
    return (nout, ntouch, over)
