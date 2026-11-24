import numpy as np
from scipy.optimize import linprog


def solve_and_certify_facilities(instance: dict) -> dict:
    """Solve capacitated facility location and return an LP branch certificate."""
    m, n = int(instance['m']), int(instance['n'])
    f = np.asarray(instance['fixed_cost'], dtype=float)
    cap = np.asarray(instance['capacity'], dtype=float)
    d = np.asarray(instance['demand'], dtype=float)
    c = np.asarray(instance['cost_matrix'], dtype=float)
    if f.shape != (m,) or cap.shape != (m,) or d.shape != (n,) or c.shape != (m, n):
        raise ValueError('inconsistent facility instance dimensions')
    if np.any(f <= 0) or np.any(cap <= 0) or np.any(d <= 0) or np.any(c < 0):
        raise ValueError('costs, capacities, and demands must meet task assumptions')
    total = float(d.sum())

    # Compact feasible incumbent from all capacity-feasible subsets.
    best = np.inf
    best_y = None
    best_x = None
    for mask in range(1 << m):
        y = np.array([(mask >> i) & 1 for i in range(m)], dtype=int)
        if float(cap @ y) + 1e-10 < total:
            continue
        # Transportation LP for a fixed opening vector.
        Aeq = np.zeros((n, m*n))
        for j in range(n):
            Aeq[j, j::n] = 1.0
        Aub = np.zeros((m, m*n))
        for i in range(m):
            Aub[i, i*n:(i+1)*n] = 1.0
        res = linprog(c.ravel(), A_ub=Aub, b_ub=cap*y, A_eq=Aeq,
                      b_eq=d, bounds=(0, None), method='highs')
        if res.success:
            val = float(f @ y + res.fun)
            if val < best:
                best, best_y, best_x = val, y.copy(), res.x.reshape(m, n)
    if best_y is None:
        raise ValueError('instance is infeasible')

    # For a partial y assignment, relax undecided indicators to [0,1].
    # Variables are (x_ij, y_i); the relaxation contains every completion,
    # hence its optimum is a sound lower bound for that entire cube.
    nv = m*n + m
    obj = np.r_[c.ravel(), f]
    Aeq = np.zeros((n, nv))
    for j in range(n):
        Aeq[j, j:m*n:n] = 1.0
    Aub = np.zeros((m, nv))
    for i in range(m):
        Aub[i, i*n:(i+1)*n] = 1.0
        Aub[i, m*n+i] = -cap[i]
    bounds_base = [(0, None)]*(m*n) + [(0, 1)]*m

    def lower_bound(closed, opened):
        if float(cap.sum() - cap[list(closed)].sum()) + 1e-10 < total:
            return np.inf
        bounds = list(bounds_base)
        for i in closed:
            bounds[m*n+i] = (0, 0)
        for i in opened:
            bounds[m*n+i] = (1, 1)
        r = linprog(obj, A_ub=Aub, b_ub=np.zeros(m), A_eq=Aeq,
                    b_eq=d, bounds=bounds, method='highs')
        return float(r.fun) if r.success else np.inf

    leaves = []
    eps = 1e-5
    def visit(closed, opened, next_i):
        lb = lower_bound(closed, opened)
        if lb >= best - eps:
            leaves.append({'fixed_closed': list(closed), 'fixed_open': list(opened), 'lower_bound': lb})
            return
        if next_i >= m:
            # A complete feasible/infeasible assignment cannot remain below the incumbent.
            raise RuntimeError('branch-and-bound reached an unpruned complete assignment')
        visit(closed + (next_i,), opened, next_i+1)
        visit(closed, opened + (next_i,), next_i+1)
    visit((), (), 0)

    # Bottom-up sibling merging preserves exact disjoint cube coverage. The
    # merged parent bound is recomputed, not inferred from child bounds.
    changed = True
    while changed:
        changed = False
        index = {}
        for k, leaf in enumerate(leaves):
            key = (tuple(leaf['fixed_closed']), tuple(leaf['fixed_open']))
            index[key] = k
        for k, leaf in enumerate(leaves):
            C, O = set(leaf['fixed_closed']), set(leaf['fixed_open'])
            for bit in sorted(C | O):
                C2, O2 = C.copy(), O.copy()
                C2.discard(bit); O2.discard(bit)
                a = (tuple(sorted(C2 | {bit})), tuple(sorted(O2)))
                b = (tuple(sorted(C2)), tuple(sorted(O2 | {bit})))
                ia, ib = index.get(a), index.get(b)
                if ia is None or ib is None or ia == ib:
                    continue
                parent_lb = lower_bound(tuple(sorted(C2)), tuple(sorted(O2)))
                if parent_lb < best - eps:
                    continue
                for q in sorted((ia, ib), reverse=True):
                    leaves.pop(q)
                leaves.append({'fixed_closed': list(sorted(C2)), 'fixed_open': list(sorted(O2)), 'lower_bound': parent_lb})
                changed = True
                break
            if changed:
                break
    if len(leaves) > 80:
        raise RuntimeError('certificate exceeds 80 leaves: %d' % len(leaves))
    return {'open_y': best_y.astype(int).tolist(), 'flows_X': best_x,
            'upper_bound': best, 'proof_leaves': leaves}
