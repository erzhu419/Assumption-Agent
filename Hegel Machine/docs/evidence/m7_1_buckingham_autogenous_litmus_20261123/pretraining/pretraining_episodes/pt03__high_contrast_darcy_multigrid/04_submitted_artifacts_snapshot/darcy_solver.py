"""Conservative cell-centered Darcy solver with Galerkin multigrid PCG."""
import numpy as np
from scipy import linalg, sparse


def _faces(k):
    a, b = k[:-1], k[1:]
    y = (2.0 * a * b) / (a + b)
    a, b = k[:, :-1], k[:, 1:]
    x = (2.0 * a * b) / (a + b)
    d = np.zeros_like(k)
    d[:-1] += y
    d[1:] += y
    d[:, :-1] += x
    d[:, 1:] += x
    d[0] += k[0]
    d[-1] += k[-1]
    d[:, 0] += k[:, 0]
    d[:, -1] += k[:, -1]
    return y, x, d


def _stencil(k, y, x, p):
    p = p.reshape(k.shape)
    vy = y * (p[:-1] - p[1:])
    vx = x * (p[:, :-1] - p[:, 1:])
    q = np.zeros_like(p)
    q[:-1] += vy
    q[1:] -= vy
    q[:, :-1] += vx
    q[:, 1:] -= vx
    q[0] += k[0] * p[0]
    q[-1] += k[-1] * p[-1]
    q[:, 0] += k[:, 0] * p[:, 0]
    q[:, -1] += k[:, -1] * p[:, -1]
    return q.ravel()


def _fine_matrix(y, x, d):
    n, m = d.shape
    horizontal = np.zeros((n, m))
    horizontal[:, :-1] = -x
    return sparse.diags((d.ravel(), horizontal.ravel()[:-1], horizontal.ravel()[:-1], -y.ravel(), -y.ravel()),
                        (0, -1, 1, -m, m), format='csr')


def _aggregates(n, row, col, weight, rng):
    labels = np.full(n, -1, dtype=np.int32)
    # Greedy stars, selected in batches of nonadjacent local maxima.
    order = rng.permutation(n).astype(np.int32)
    count = 0
    remaining = np.arange(n, dtype=np.int32)
    while remaining.size:
        high = order.copy()
        np.maximum.at(high, row, order[col])
        roots = remaining[high[remaining] == order[remaining]]
        labels[roots] = np.arange(count, count + roots.size, dtype=np.int32)
        count += roots.size
        eligible = (labels[row] < 0) & (labels[col] >= 0)
        i, j, w = row[eligible], col[eligible], weight[eligible]
        best = np.zeros(n)
        np.maximum.at(best, i, w)
        pick = best[i] == w
        labels[i[pick]] = labels[j[pick]]
        remaining = remaining[labels[remaining] < 0]
        mask = (labels[row] < 0) & (labels[col] < 0)
        row, col, weight = row[mask], col[mask], weight[mask]
    return labels, count


def _hierarchy(a):
    levels = []
    rng = np.random.default_rng(12)
    while a.shape[0] > 64:
        n = a.shape[0]
        diag = a.diagonal()
        c = a.tocoo()
        off = c.row != c.col
        i, j, v = c.row[off], c.col[off], c.data[off]
        w = np.abs(v) / np.sqrt(diag[i] * diag[j])
        strong = w >= 0.08
        labels, nc = _aggregates(n, i[strong], j[strong], w[strong], rng)
        if nc > 0.76 * n:
            mask = w > 0
            labels, nc = _aggregates(n, i[mask], j[mask], w[mask], rng)
        ii, jj, vv = i[strong], j[strong], v[strong]
        weak_sum = np.bincount(i[~strong], weights=v[~strong], minlength=n)
        df = diag + weak_sum
        df = np.maximum(df, 0.5 * diag)
        # A strength-filtered Jacobi step smooths the aggregate indicators.
        p = sparse.csr_matrix((np.r_[1.0 - (2.0 / 3.0) * (diag + weak_sum) / df,
                                     -(2.0 / 3.0) * vv / df[ii]],
                               (np.r_[np.arange(n), ii], np.r_[labels, labels[jj]])),
                              shape=(n, nc))
        p.eliminate_zeros()
        pt = p.T.tocsr()
        # Bounds for the diagonal-scaled spectrum give a symmetric smoother.
        bound = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / diag)
        levels.append((a, pt, p, 1.0 / diag, 1.0 / (bound * 0.55), 1.0 / bound))
        a = (pt @ (a @ p)).tocsr()
        a = ((a + a.T) * 0.5).tocsr()
    return levels, linalg.cho_factor(a.toarray(), check_finite=False)


def _cycle(levels, coarse, r, lev=0):
    if lev == len(levels):
        return linalg.cho_solve(coarse, r, check_finite=False)
    a, pt, p, dinv, hi, lo = levels[lev]
    z = (hi * dinv) * r
    z += (lo * dinv) * (r - a @ z)
    z += p @ _cycle(levels, coarse, pt @ (r - a @ z), lev + 1)
    z += (lo * dinv) * (r - a @ z)
    z += (hi * dinv) * (r - a @ z)
    return z


def _dot(x, y):
    return np.einsum('i,i->', x, y)


def _pcg(k, fy, fx, levels, coarse, r, target):
    """Solve for a correction from a supplied residual."""
    x = np.zeros_like(r)
    if _dot(r, r) <= target:
        return x
    z = _cycle(levels, coarse, r)
    direction = z.copy()
    rz = _dot(r, z)
    for _ in range(450):
        ad = _stencil(k, fy, fx, direction)
        step = rz / _dot(direction, ad)
        x += step * direction
        r -= step * ad
        if _dot(r, r) <= target:
            break
        z = _cycle(levels, coarse, r)
        new_rz = _dot(r, z)
        direction *= new_rz / rz
        direction += z
        rz = new_rz
    return x


def solve_darcy_pressure(permeability_2d: np.ndarray, source_rhs: np.ndarray,
                         h: float, tol: float = 1e-8) -> np.ndarray:
    """Solve the harmonic-face flux balance with zero-valued ghost cells."""
    k = np.asarray(permeability_2d, dtype=np.float64)
    b = np.ascontiguousarray(source_rhs, dtype=np.float64).ravel() * (h * h)
    b2 = _dot(b, b)
    if b2 == 0:
        return np.zeros_like(k)
    fy, fx, d = _faces(k)
    levels, coarse = _hierarchy(_fine_matrix(fy, fx, d))
    target = (0.85 * tol) ** 2 * b2
    x = _pcg(k, fy, fx, levels, coarse, b.copy(), target)
    r = b - _stencil(k, fy, fx, x)
    if _dot(r, r) <= target:
        return x.reshape(k.shape)

    # Across strong faces, rounding nearly equal pressures can dominate the
    # flux error. Keep the pressure and true residual in extended precision
    # when needed, but reuse the double-precision V-cycle for corrections.
    x = x.astype(np.longdouble)
    bw = b.astype(np.longdouble)
    for _ in range(4):
        r = np.asarray(bw - _stencil(k, fy, fx, x), dtype=np.float64)
        r2 = _dot(r, r)
        if r2 <= target:
            break
        x += _pcg(k, fy, fx, levels, coarse, r, max(0.04 * target, 1e-4 * r2))
    xf = x.astype(np.float64)
    r = b - _stencil(k, fy, fx, xf)
    return (xf if _dot(r, r) <= target else x).reshape(k.shape)
