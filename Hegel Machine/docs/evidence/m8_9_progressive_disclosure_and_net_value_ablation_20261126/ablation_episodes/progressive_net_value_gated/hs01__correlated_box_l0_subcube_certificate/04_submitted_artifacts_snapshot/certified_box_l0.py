"""Global boxed L0 regression using a disjunctive root certificate."""
import numpy as np
from scipy.optimize import lsq_linear


def solve_certified_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    ridge = float(problem['ridge'])
    penalty = float(problem['l0_penalty'])
    box = float(problem['box'])
    tol = float(problem['gap_tol'])
    n, p = X.shape
    small = min(1e-10, box)
    H = X.T @ X + ridge * np.eye(p)
    c = X.T @ y
    yy = 0.5 * (y @ y)
    u = np.finfo(float).eps
    abs_X = np.abs(X)
    abs_y = np.abs(y)
    # Conservative dot-product and summation allowance for the dual witness.
    roundoff = 8 * (n + p + 8) * u
    bits = 1 << np.arange(p)
    eye = np.eye(p)
    # Used only to propose centers, never in the certified objective or dual.
    shift = 64 * u * max(1.0, np.max(H, initial=0.0))
    best = np.zeros(p)
    upper = float(yy)
    lower = np.full(1 << p, -np.inf)
    pending = []
    rowdot = lambda a, b: np.einsum('ij,ij->i', a, b)

    def gradient(w):
        return np.einsum('ij,kj->ik', w, H) - c

    def solve_face(free, fixed):
        # Disjoint faces are solved together. Rows and columns outside the
        # free set form an identity block with zero right-hand side.
        system = H * (free[:, :, None] & free[:, None, :])
        system += (np.where(free, shift, 1.0))[:, :, None] * eye
        rhs = -gradient(fixed) * free
        # Small positive-definite systems: batch the triangular substitutions
        # across supports rather than dispatching a general solve per support.
        try:
            L = np.linalg.cholesky(system)
        except np.linalg.LinAlgError:
            return fixed + np.linalg.solve(system, rhs[..., None])[..., 0]
        v = rhs.copy()
        for j in range(p):
            v[:, j] = (v[:, j] - rowdot(L[:, j, :j], v[:, :j])) / L[:, j, j]
        for j in range(p - 1, -1, -1):
            v[:, j] = (v[:, j] - rowdot(L[:, j+1:, j], v[:, j+1:])) / L[:, j, j]
        return fixed + v

    def evaluate(ids, radius, charge, w, candidate=True):
        nonlocal upper, best
        # The computed residual a is a free dual witness; its accuracy as
        # a residual affects sharpness, not validity. Apply Fenchel's
        # inequalities to the loss and ridge, then minimize the resulting
        # affine function over the support box.
        a = np.einsum('ij,kj->ik', w, X) - y
        g = np.einsum('ij,jk->ik', a, X) + ridge * w
        loss_norm = rowdot(a, a)
        ridge_norm = ridge * rowdot(w, w)
        const = -np.einsum('ij,j->i', a, y) - 0.5 * (loss_norm + ridge_norm)
        lb = const + charge - rowdot(radius, np.abs(g))
        scale = (np.einsum('ij,j->i', np.abs(a), abs_y)
                 + 0.5 * (loss_norm + ridge_norm) + charge
                 + rowdot(radius, np.einsum('ij,jk->ik', np.abs(a), abs_X)
                          + ridge * np.abs(w)))
        lower[ids] = np.maximum(lower[ids], lb - roundoff * scale)
        if candidate:
            values = (0.5 * (loss_norm + ridge_norm)
                      + penalty * np.count_nonzero(np.abs(w) > 1e-10, axis=1))
            v = w[int(np.argmin(values))]
            r = X @ v - y
            value = float(0.5 * (r @ r + ridge * (v @ v))
                          + penalty * np.count_nonzero(np.abs(v) > 1e-10))
            if value < upper:
                upper, best = value, v.copy()

    # A support charges its designated active coordinates but lets them be
    # zero. Its other coordinates have radius 1e-10 to match the objective's
    # activation threshold. This also bounds the stricter exact-zero domain.
    batch = 512
    if p:
        for start in range(0, 1 << p, batch):
            ids = np.arange(start, min(start + batch, 1 << p))
            active = (ids[:, None] & bits) != 0
            radius = np.where(active, box, small)
            charge = penalty * active.sum(axis=1)
            center = solve_face(active, np.zeros_like(radius))
            evaluate(ids, radius, charge, np.clip(center, -4 * box, 4 * box), False)
            w = np.clip(center, -radius, radius)
            w = np.where(active, w, -small * np.sign(gradient(w)))
            evaluate(ids, radius, charge, w)
            if np.any(lower[ids] < upper - tol):
                pending.append((ids, active, radius, charge, w))
    else:
        lower[0] = yy - roundoff * yy

    for ids, active, radius, charge, w in pending:
        for _ in range(8):
            keep = lower[ids] < upper - tol
            if not np.any(keep):
                break
            ids, active, radius, charge, w = (a[keep] for a in (ids, active, radius, charge, w))
            g = gradient(w)
            pinned = ((w <= -box) & (g >= 0)) | ((w >= box) & (g <= 0))
            free = active & ~pinned
            fixed = np.where(free, 0.0, np.where(active, w, -small * np.sign(g)))
            w = np.clip(solve_face(free, fixed), -radius, radius)
            evaluate(ids, radius, charge, w)
        # BVLS resolves supports whose projected Newton iterates still have
        # an ambiguous active face. Its stopping report is not a certificate.
        for j in np.flatnonzero(lower[ids] < upper - tol):
            if lower[ids[j]] >= upper - tol:
                continue
            a = active[j]
            v = w[j].copy()
            if np.any(a):
                design = X[:, a]
                if ridge:
                    design = np.vstack((design, np.sqrt(ridge) * np.eye(a.sum())))
                for _ in range(3):
                    target = y - X[:, ~a] @ v[~a]
                    if ridge:
                        target = np.r_[target, np.zeros(a.sum())]
                    fit = lsq_linear(design, target, bounds=(-box, box),
                                     method='bvls', tol=1e-13)
                    v[a] = np.clip(fit.x, -box, box)
                    v[~a] = -small * np.sign(gradient(v[None])[0, ~a])
                    evaluate(ids[j:j+1], radius[j:j+1], charge[j:j+1], v[None])
                    if lower[ids[j]] >= upper - tol:
                        break

    bound = min(upper, float(np.min(lower)))
    if not 0 <= upper - bound <= tol:
        raise ArithmeticError(f'Unable to close certificate gap: {upper - bound:g}')
    return {'weights': best, 'objective': upper, 'certified_lower_bound': bound,
            'certificate_leaves': [{'fixed_zero': [], 'fixed_nonzero': [],
                                    'leaf_lower_bound': bound}]}
