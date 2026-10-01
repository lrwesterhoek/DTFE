"""Exact fields of the crossed Zel'dovich waves of tests/generate_ps_test_data.py --crossed-waves.

Each axis maps independently: x = q + A sin(kq), k = 2 pi / L, A = F / k, v = 100 A sin(kq).
Everything factorises over the axes:
  rho/rho_bar (x, y, z) = n(x) n(y) n(z),   n(x) = sum over roots q of 1 / |1 + F cos kq|
  stream count          = c(x) c(y) c(z),   c = number of roots (1 or 3 for 1 < F < 4.6)
  mass-weighted mean v_x = sum v(q)/|J| / n(x)   (the y and z factors cancel)
and so do the cell averages, which come in closed form from the inverse of each monotone branch:
  cell mean of n over [a, b)        = (Lagrangian length mapped into [a, b)) / (b - a)
  cell mass-weighted mean of v_x    = integral of v dq over that length / that length
  cell mean stream count            = (x-length of every branch image inside [a, b)) / (b - a)
"""
from __future__ import annotations

import numpy as np


class Wave1D:
    def __init__(self, L: float, F: float):
        self.L, self.F = float(L), float(F)
        self.k = 2 * np.pi / self.L
        self.A = self.F / self.k
        if self.F > 1:
            t = np.arccos(-1.0 / self.F)
            self.q1, self.q2 = t / self.k, (2 * np.pi - t) / self.k          # dx/dq = 0 (the folds)
            # increasing branch over [q2 - L, q1] (unwrapped), decreasing branch over [q1, q2]
            self.branches = [(self.q2 - self.L, self.q1, +1), (self.q1, self.q2, -1)]
        else:
            self.branches = [(0.0, self.L, +1)]

    def x(self, q):
        return q + self.A * np.sin(self.k * q)

    def J(self, q):
        return 1 + self.F * np.cos(self.k * q)

    def v(self, q):
        return 100.0 * self.A * np.sin(self.k * q)

    def V(self, q):                                   # antiderivative of v in q
        return -100.0 * self.A * np.cos(self.k * q) / self.k

    def _inverse(self, y, lo, hi, sign):
        """q in [lo, hi] with x(q) = y on a monotone branch (bisection to machine precision)."""
        a = np.full(np.shape(y), lo, float)
        b = np.full(np.shape(y), hi, float)
        for _ in range(64):
            m = 0.5 * (a + b)
            below = (self.x(m) < y) if sign > 0 else (self.x(m) > y)
            a = np.where(below, m, a)
            b = np.where(below, b, m)
        return 0.5 * (a + b)

    def roots(self, xv):
        """All roots per point: (q, valid) arrays of shape (len(x), 6)."""
        xv = np.asarray(xv, float) % self.L
        qs, ok = [], []
        for lo, hi, sign in self.branches:
            xa, xb = sorted((self.x(lo), self.x(hi)))
            for s in (-self.L, 0.0, self.L):
                y = xv + s
                inside = (y >= xa) & (y < xb)
                qs.append(np.where(inside, self._inverse(np.clip(y, xa, xb), lo, hi, sign), 0.0))
                ok.append(inside)
        return np.stack(qs, 1), np.stack(ok, 1)

    def point(self, xv):
        """n, count, mass-weighted mean v at points."""
        q, ok = self.roots(xv)
        w = np.where(ok, 1.0 / np.abs(self.J(q)), 0.0)
        n = w.sum(1)
        vbar = (w * self.v(q)).sum(1) / n
        return n, ok.sum(1), vbar

    def cells(self, edges):
        """Per cell [e_i, e_{i+1}): mean n, mean stream count, mass-weighted mean v."""
        a, b = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
        mass, vint, xlen = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
        for lo, hi, sign in self.branches:
            xa, xb = sorted((self.x(lo), self.x(hi)))
            for s in (-self.L, 0.0, self.L):
                ya, yb = np.clip(a + s, xa, xb), np.clip(b + s, xa, xb)
                has = yb > ya
                qa = self._inverse(ya, lo, hi, sign)
                qb = self._inverse(yb, lo, hi, sign)
                mass += np.where(has, np.abs(qb - qa), 0.0)
                vint += np.where(has, np.abs(self.V(qb) - self.V(qa)) * np.sign(self.V(qb) - self.V(qa))
                                 * np.sign(qb - qa), 0.0)
                xlen += np.where(has, yb - ya, 0.0)
        width = b - a
        return mass / width, xlen / width, vint / np.where(mass > 0, mass, 1.0)


def slice_truth(L, F, n_pix, z0):
    """Exact point fields on an n_pix^2 slice at z = z0 (pixel centres), arrays [ix, iy]."""
    w = Wave1D(L, F)
    c = (np.arange(n_pix) + 0.5) * L / n_pix
    n1, c1, v1 = w.point(c)
    nz, cz, _ = w.point(np.array([z0]))
    rho = n1[:, None] * n1[None, :] * nz[0]
    streams = c1[:, None] * c1[None, :] * cz[0]
    vx = np.repeat(v1[:, None], n_pix, 1)
    vy = np.repeat(v1[None, :], n_pix, 0)
    return {"rho": rho, "streams": streams, "vx": vx, "vy": vy}


def grid_truth(L, F, n):
    """Exact CELL-AVERAGED fields on an n^3 grid, returned as 1-D factors (the 3-D field is their
    outer product): rho = n1[i] n1[j] n1[k], streams = s1[i] s1[j] s1[k], v_x = v1[i]."""
    w = Wave1D(L, F)
    edges = np.linspace(0, L, n + 1)
    return w.cells(edges)


if __name__ == "__main__":
    # self-checks: mass conservation, the analytic stream counts, and cell averages vs fine sampling
    w = Wave1D(100.0, 1.8)
    n, c, v = w.point(np.linspace(0, 100, 20001)[:-1])
    print("mean n", n.mean(), "(1 up to the caustic singularities)  counts", sorted(set(c.tolist())))
    m, s, vb = w.cells(np.linspace(0, 100, 257))
    print("cells: mean n", m.mean(), " mean count", s.mean(), " max count", s.max())
    q = np.linspace(0, 100, 4_000_001)[:-1]
    x = w.x(q) % 100
    idx = np.floor(x / (100 / 256)).astype(int)
    hist = np.bincount(idx, minlength=256) / (len(q) / 256)
    vh = np.bincount(idx, weights=w.v(q), minlength=256) / np.maximum(np.bincount(idx, minlength=256), 1)
    print("cell n vs sampled: max rel", np.abs(m / hist - 1).max(), "  v max abs diff", np.abs(vb - vh).max())
