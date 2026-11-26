"""Box-constrained sparse ridge with replayable subcube certificates."""
import numpy as np


def _diagonal(G):
    """Choose a PSD minorant per Gram block, accounting for column scale."""
    p = len(G)
    d = np.zeros(p)
    unseen = set(range(p))
    while unseen:
        block = [unseen.pop()]
        for j in block:
            near = [k for k in unseen if G[j, k] != 0]
            block.extend(near)
            unseen.difference_update(near)
        B = G[np.ix_(block, block)]
        margin = 64 * np.finfo(float).eps * len(B) * np.max(np.abs(B))
        off = np.sum(np.abs(B - np.diag(B.diagonal())), axis=1)
        gersch = np.maximum(0, B.diagonal() - off - margin)
        v = np.linalg.eigvalsh(B)[0]
        if v <= margin:
            # A singular block can forbid positive curvature on a column
            # even when that row is individually diagonally dominant.
            continue
        shift = np.full(len(B), v - margin)
        slack = np.linalg.eigvalsh(B - np.diag(gersch))[0] - margin
        # Do not clip negative slack to zero and keep an invalid candidate.
        alt = gersch + slack if slack >= 0 else shift
        candidate = alt if alt.sum() > shift.sum() else shift
        # If R - floor*I is PSD, diagonal congruence gives a PSD
        # residual in the original coordinates without rescaling the box.
        norms_sq = B.diagonal()
        scale = np.sqrt(norms_sq)
        R = B / scale[:, None] / scale[None, :]
        relative_margin = 64 * np.finfo(float).eps * len(B) * max(1., np.max(np.abs(R)))
        floor = np.linalg.eigvalsh(R)[0] - relative_margin
        if floor > np.mean(candidate / norms_sq):
            candidate = floor * norms_sq
        d[block] = candidate
    return d


def solve_proof_carrying_box_l0(problem: dict) -> dict:
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l0, l2, M = (float(problem[k]) for k in ('lam0', 'lam2', 'M'))
    p = X.shape[1]
    G = X.T @ X
    h = X.T @ y
    d = _diagonal(G)
    H = G - np.diag(d)
    q = l2 + d
    diag = G.diagonal() + l2
    a = np.minimum(M, np.sqrt(2*l0/q))
    slope = l0/a + 0.5*q*a
    zero = np.zeros(p, dtype=bool)
    tol = 6e-6
    pad = 32*np.finfo(float).eps * (1 + np.dot(y,y) + M*np.sum(np.abs(h)) + M*M*np.sum(np.abs(G)) + p*(l0 + l2*M*M))

    def cost(w):
        r = X @ w - y
        return float(0.5*(r @ r + l2*(w @ w)) + l0*np.count_nonzero(np.abs(w) > 1e-8))

    best = np.zeros(p)
    ub = cost(best)

    def polish(mask, seed):
        nonlocal best, ub
        j = np.flatnonzero(mask)
        if not len(j):
            return
        A = G[np.ix_(j,j)] + l2*np.eye(len(j))
        b = h[j]
        v = np.linalg.solve(A,b)
        if np.any(np.abs(v)>M):
            if len(j) > 1:
                from scipy.optimize import minimize
                init = np.clip(seed[j], -M,M)
                def fun(w):
                    Aw = A @ w
                    return 0.5*(w @ Aw) - b @ w, Aw-b
                v = minimize(fun, init, jac=True, bounds=[(-M,M)]*len(j),
                             method='L-BFGS-B', options={'gtol': 2e-12, 'ftol': 1e-15}).x
                # Polish the free coordinates of the returned box face.
                free = np.abs(v) < M
                if np.any(free):
                    trial = v.copy()
                    trial[free] = np.linalg.solve(A[np.ix_(free,free)], b[free]-A[np.ix_(free,~free)]@v[~free])
                    if np.max(np.abs(trial)) <= M and fun(trial)[0] <= fun(v)[0] + 1e-13:
                        v = trial
            v = np.clip(v,-M,M)
        w = np.zeros(p)
        w[j] = v
        w[np.abs(w) <= 1e-8] = 0.
        f = cost(w)
        if f < ub:
            best, ub = w, f

    def certificate(c, z, act):
        r = y - X @ c
        g = np.abs(X.T @ r + d*c)
        b = np.minimum(M, g/q)
        delta = l0 - b*(g - 0.5*q*b)
        # Also bound the unpenalized interval in the stated 1e-8 support
        # convention. This only lowers the replayable exact-zero witness.
        small = np.minimum(b, 1e-8)
        inactive = -small*(g - 0.5*q*small)
        free = ~(z|act)
        lb = 0.5*(r @ r) + r @ (X @ c) + 0.5*((d*c) @ c)
        lb += delta[act].sum() + inactive[z].sum()
        lb += np.minimum(inactive[free], delta[free]).sum()
        return float(lb-pad)

    def relax(z, act, seed, sweeps):
        c = seed.copy()
        c[z] = 0.
        ids = np.flatnonzero(~z)
        hc = H @ c
        hd = H.diagonal()
        for _ in range(sweeps):
            change = 0.
            for j in ids:
                u = h[j] - hc[j] + hd[j]*c[j]
                uabs = abs(u)
                if act[j]:
                    val = min(M, uabs/diag[j])
                elif uabs <= slope[j]:
                    val = 0.
                elif uabs <= slope[j] + hd[j]*a[j] and hd[j] > 0.:
                    val = (uabs-slope[j])/hd[j]
                else:
                    val = min(M, max(a[j],uabs/diag[j]))
                x = np.copysign(val,u)
                dv = x-c[j]
                if dv:
                    c[j] = x
                    hc += dv*H[:,j]
                    change = max(change,abs(dv))
            if change <= 2e-12 * max(1.,M):
                break
        return c

    root = relax(zero, zero, best, 512)
    for t in (0., .25, .5, .75, 1.):
        polish(np.abs(root) > np.maximum(1e-8,t*a),root)
    leaves = []
    stack = [(zero, zero, root)]
    while stack:
        z, act, seed = stack.pop()
        # A parent witness can already close its child.
        lb = certificate(seed,z,act)
        c = seed
        if lb < ub - tol:
            c = relax(z,act,seed,192)
            lb = certificate(c,z,act)
        if lb < ub - tol:
            polish(act | (~z & (np.abs(c) >= .5*a)), c)
        if lb < ub - tol:
            free = np.flatnonzero(~(z|act))
            if len(free) == 0:
                c = relax(z,act,best,4096)
                lb = certificate(c,z,act)
                polish(act,c)
                if lb < ub - 9.9e-6:
                    raise RuntimeError('Fully fixed cube did not converge')
            else:
                f = np.clip(np.abs(c[free])/a[free],0,1)
                score = f*(1-f)*(q[free]*a[free]**2)
                k = free[np.argmax(score)]
                az, aa = z.copy(), act.copy()
                az[k] = aa[k] = True
                if abs(c[k]) > .5*a[k]:
                    stack.extend(((az,act,c),(z,aa,c)))
                else:
                    stack.extend(((z,aa,c),(az,act,c)))
                continue
        leaves.append({'fixed_zero': z, 'fixed_active': act, 'lb': lb,
                       'minorant_diagonal': d, 'minorant_center': c})
        if len(leaves) + len(stack) > 2048:
            raise RuntimeError('Certificate exceeds the 2048-leaf budget')
    lower = min(ub, min(leaf['lb'] for leaf in leaves))
    if len(leaves)>2048 or ub-lower>1e-5:
        raise RuntimeError(f'Certificate exceeds contract: {len(leaves)} leaves, gap {ub-lower:g}')
    return {'weights': best, 'objective': ub, 'certified_lower_bound': lower,
            'subcube_leaves': leaves}
