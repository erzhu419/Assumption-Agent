"""Infer canal parameters by matching downstream gauges to a finite-volume model."""

import numpy as np


def _simulate(heights, modes, n, steps, dt, dx, times, cells):
    """Advance a batch of canals and retain only the requested samples."""
    g = 9.81
    ratio = dt / dx
    x = (np.arange(n) + 0.5) * dx
    bed = np.asarray(heights)[:, None] * np.exp(-0.5 * ((x - 400.) / 45.) ** 2)
    # Each interface sees the cell on each side, including reflected ghosts.
    left = np.r_[0, np.arange(n)]
    right = np.r_[np.arange(n), n - 1]
    crest = np.maximum(bed[:, left], bed[:, right])
    h = 2. - bed
    h[:, x < 80.] = np.where(np.asarray(modes), 2.65, 1.45)[:, None] - bed[:, x < 80.]
    q = np.zeros_like(h)
    record = np.empty((len(h), len(times), len(cells)))
    rows = [[] for _ in range(steps)]
    for row, t in enumerate(times):
        rows[t].append(row)
    max_time = int(max(times))
    for t in range(max_time + 1):
        eta = h + bed
        for row in rows[t]:
            record[:, row, :] = eta[:, cells]
        if t == max_time:
            break
        u = q / np.maximum(h, 1.e-9)
        ul, ur = u[:, left], u[:, right]
        ul[:, 0] *= -1.
        ur[:, -1] *= -1.
        hl = np.maximum(0., eta[:, left] - crest)
        hr = np.maximum(0., eta[:, right] - crest)
        ql, qr = hl * ul, hr * ur
        speed = np.maximum(1.e-6, np.maximum(np.abs(ul) + np.sqrt(g * hl),
                                             np.abs(ur) + np.sqrt(g * hr)))
        pl, pr = (0.5 * g) * hl**2, (0.5 * g) * hr**2
        f = 0.5 * (ql + qr - speed * (hr - hl))
        fq = 0.5 * (ql * ul + pl + qr * ur + pr - speed * (qr - ql))
        h -= ratio * (f[:, 1:] - f[:, :-1])
        q += ratio * (-(fq[:, 1:] - fq[:, :-1]) + pl[:, 1:] - pr[:, :-1])
    return record


def invert_canal_regime(obs_bundle: dict) -> dict:
    """Fit both release modes and return the best full free-surface trajectory."""
    n, steps = obs_bundle['N'], obs_bundle['T_steps']
    dt, dx = obs_bundle['dt'], obs_bundle['dx']
    times = np.asarray(obs_bundle['obs_step_indices'], dtype=int)
    cells = np.asarray(obs_bundle['gauge_cells'], dtype=int)
    obs = np.asarray(obs_bundle['eta_obs'], dtype=float)

    def loss(heights, modes):
        pred = _simulate(heights, modes, n, steps, dt, dx, times, cells)
        return np.sum((pred - obs)**2, axis=(1, 2))

    # Search globally in the two discrete modes, then refine around the winner.
    grid = np.linspace(0., 0.75, 31)
    heights = np.tile(grid, 2)
    modes = np.repeat([False, True], len(grid))
    costs = loss(heights, modes)
    k = int(np.argmin(costs))
    mode = bool(modes[k])
    i = k % len(grid)
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    best_height, best_cost = heights[k], costs[k]
    # Batched refinements keep the search dependency-free.
    for _ in range(3):
        points = np.linspace(lo, hi, 9)
        costs = loss(points, np.full(len(points), mode))
        j = int(np.argmin(costs))
        if costs[j] < best_cost:
            best_height, best_cost = points[j], costs[j]
        lo, hi = points[max(0, j - 1)], points[min(len(points) - 1, j + 1)]

    eta = _simulate([best_height], [mode], n, steps, dt, dx,
                    np.arange(steps), np.arange(n))[0]
    return {'sluice_high_mode': mode, 'sill_height_m': float(best_height), 'eta_full': eta}
