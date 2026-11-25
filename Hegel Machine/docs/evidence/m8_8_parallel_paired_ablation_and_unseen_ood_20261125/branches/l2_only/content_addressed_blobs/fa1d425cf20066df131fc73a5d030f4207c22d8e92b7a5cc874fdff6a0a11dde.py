"""Certified capacitated facility location with splittable customer demand."""

import numpy as np
from scipy.optimize import linprog


_CERT_TOL = 1e-5


def _capacity_feasible(capacity, demand):
    """Compare subset capacities and total demand exactly for binary64 inputs."""
    m = len(capacity)
    ratios = [float(x).as_integer_ratio() for x in (*capacity, *demand)]
    denominator = max(q for _, q in ratios)
    units = [p * (denominator // q) for p, q in ratios]
    required = sum(units[m:])
    totals = [0]
    for amount in units[:m]:
        totals += [total + amount for total in totals]
    return np.fromiter((total >= required for total in totals), bool, len(totals))


def _compact(bounds, minimum):
    """Merge siblings when their parent is certified by the smaller bound."""
    m = len(bounds).bit_length() - 1
    leaves = [{'fixed_closed': [i for i in range(m) if not (mask >> i & 1)],
               'fixed_open': [i for i in range(m) if mask >> i & 1],
               'lower_bound': float(bound)} for mask, bound in enumerate(bounds)]
    for i in range(m):
        parents = []
        for left, right in zip(leaves[::2], leaves[1::2]):
            bound = min(left['lower_bound'], right['lower_bound'])
            if bound < minimum:
                raise RuntimeError('Transportation duals did not close the certificate gap')
            parents.append({'fixed_closed': left['fixed_closed'][1:],
                            'fixed_open': right['fixed_open'][1:],
                            'lower_bound': bound})
        leaves = parents
    return leaves


def solve_and_certify_facilities(instance: dict) -> dict:
    """Return an optimal opening/flow pair and a compact lower-bound proof.

    For a customer price vector p, shipping costs are bounded below by
    p @ demand - sum_i capacity[i] * max(0, max_j(p[j]-cost[i,j])) * y[i].
    Each price contributes a valid affine cut for all opening patterns. For
    a leaf, minimize the pointwise maximum of the cuts plus opening costs
    over its capacity-feasible binary patterns. This is a small epigraph
    capacity-knapsack bound. Taking the smaller sibling bound implements
    this same minimization on the parent, without dropping any cuts.
    """
    f, a, d, c = (np.asarray(instance[key], dtype=float) for key in
                  ('fixed_cost', 'capacity', 'demand', 'cost_matrix'))
    m, n = c.shape
    feasible = _capacity_feasible(a, d)
    if not feasible[-1]:
        raise ValueError('Total facility capacity is below total demand')

    masks = np.arange(1 << m)
    indicators = ((masks[:, None] >> np.arange(m)) & 1).astype(np.longdouble)
    fwide, awide, dwide, cwide = (v.astype(np.longdouble) for v in (f, a, d, c))
    fixed = indicators @ fwide
    width = np.finfo(np.longdouble).eps

    # No facility can ship more than total demand. Round that total
    # upward so tightening the row bounds cannot remove feasible flows.
    demand_upper = np.longdouble(0)
    for quantity in dwide:
        demand_upper = np.nextafter(demand_upper + quantity, np.longdouble(np.inf))
    usable = np.minimum(awide, demand_upper)

    def cut(price):
        p = np.asarray(price, dtype=np.longdouble)
        q = np.maximum(0, (p - cwide).max(axis=1))
        penalty = usable * q
        value = p @ dwide + indicators @ (fwide - penalty)
        # The positive-part maximum inherits the relative error of the
        # differences p[j] - cost[i,j]. Only open facilities contribute
        # opening costs or capacity penalties to a given cut value.
        scale = np.abs(p) @ dwide + fixed + indicators @ penalty
        return value - (4 * (m + n + 4) * width) * scale

    lower = cut(c.min(axis=0))
    lower[~feasible] = np.inf
    pending = lower.copy()
    upper = np.inf
    best_mask = None
    best_flows = None
    # Express the LP in units of a typical demand. A power of two
    # preserves binary inputs and keeps the solver tolerances independent
    # of the physical unit used for flows.
    flow_exp = int(np.frexp(np.sort(d)[n // 2])[1])
    cap_lp = np.nextafter(float(demand_upper), np.inf)
    a_lp = np.ldexp(np.minimum(a, cap_lp), -flow_exp)
    d_lp = np.ldexp(d, -flow_exp)
    c_lp = np.ldexp(c, flow_exp)
    eq = np.tile(np.eye(n), (1, m))
    ub = np.kron(np.eye(m), np.ones((1, n)))
    lp_options = {'primal_feasibility_tolerance': 1e-10,
                  'dual_feasibility_tolerance': 1e-10}

    while True:
        mask = int(np.argmin(pending))
        if pending[mask] >= upper:
            break
        rows = np.flatnonzero(indicators[mask])
        cols = (rows[:, None] * n + np.arange(n)).ravel()
        lp = linprog(c_lp[rows].ravel(), A_ub=ub[:len(rows), :len(cols)],
                     b_ub=a_lp[rows], A_eq=eq[:, :len(cols)], b_eq=d_lp,
                     method='highs-ds', options=lp_options)
        if not lp.success:
            raise RuntimeError(f'Transportation LP failed for opening pattern {mask}: {lp.message}')
        flows = np.ldexp(lp.x.reshape(len(rows), n), flow_exp)
        objective = fixed[mask] + np.sum(cwide[rows] * flows, dtype=np.longdouble)
        if objective < upper:
            upper, best_mask = objective, mask
            best_flows = np.zeros_like(c)
            best_flows[rows] = flows
        lower = np.maximum(lower, cut(np.ldexp(lp.eqlin.marginals.astype(np.longdouble), -flow_exp)))
        pending = np.maximum(pending, lower)
        pending[mask] = np.inf

    rounded_lower = np.nextafter(lower.astype(float), -np.inf, where=np.isfinite(lower), out=lower.astype(float))
    upper = float(upper)
    leaves = _compact(rounded_lower, upper - _CERT_TOL)
    return {'open_y': indicators[best_mask].astype(int).tolist(),
            'flows_X': best_flows, 'upper_bound': upper, 'proof_leaves': leaves}
