"""Branch-and-bound certificate for splittable capacitated facility location."""
from heapq import heappop, heappush
from math import inf, nextafter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, diags, eye, hstack, kron, vstack


_TOL = 9.5e-6  # Keep a margin inside the requested 1e-5.
_LP_OPTIONS = {'primal_feasibility_tolerance': 1e-9,
               'dual_feasibility_tolerance': 1e-9}


def _dyadic(values):
    terms = [float(v).as_integer_ratio() for v in values]
    den = max(q for _, q in terms)
    return [p * (den // q) for p, q in terms], den


def _transport(ids, capacities, demands, costs):
    """Successive shortest augmenting paths in exact dyadic coordinates."""
    k, n = len(ids), len(demands)
    source, sink = k + n, k + n + 1
    graph = [[] for _ in range(sink + 1)]

    def arc(a, b, cap, price):
        forward = [b, len(graph[b]), cap, price]
        reverse = [a, len(graph[a]), 0, -price]
        graph[a].append(forward)
        graph[b].append(reverse)
        return reverse

    for row, i in enumerate(ids):
        arc(source, row, capacities[i], 0)
    remaining = sum(demands)
    arcs = [[arc(row, k + j, demands[j], costs[i][j]) for j in range(n)]
            for row, i in enumerate(ids)]
    for j, d in enumerate(demands):
        arc(k + j, sink, d, 0)

    potential = [0] * len(graph)
    while remaining:
        dist = [inf] * len(graph)
        prev = [None] * len(graph)
        dist[source] = 0
        queue = [(0, source)]
        while queue:
            value, a = heappop(queue)
            if value != dist[a]:
                continue
            if a == sink:
                break
            for e in graph[a]:
                b, _, cap, price = e
                if cap:
                    trial = value + price + potential[a] - potential[b]
                    if trial < dist[b]:
                        dist[b] = trial
                        prev[b] = (a, e)
                        heappush(queue, (trial, b))
        # Truncating distances at the sink preserves nonnegative reduced costs,
        # including arcs incident to vertices not settled by the early stop.
        for a in range(len(graph)):
            potential[a] += min(dist[a], dist[sink])
        b, amount = sink, remaining
        while b != source:
            b, e = prev[b]
            amount = min(amount, e[2])
        b = sink
        while b != source:
            a, e = prev[b]
            e[2] -= amount
            graph[b][e[1]][2] += amount
            b = a
        remaining -= amount

    flow = [[e[2] for e in row] for row in arcs]
    value = sum(amount * price for i, row in zip(ids, flow)
                for amount, price in zip(row, costs[i]))
    return value, flow


def _capacity_table(capacity, total):
    # Binary64 data are dyadic: put all amounts over a common denominator so
    # that an infeasibility leaf never rests on a rounded capacity comparison.
    m = len(capacity)
    capacities = [0] * (1 << m)
    for mask in range(1, len(capacities)):
        bit = mask & -mask
        capacities[mask] = capacities[mask ^ bit] + capacity[bit.bit_length() - 1]
    return np.array([v >= total for v in capacities])


def solve_and_certify_facilities(instance: dict) -> dict:
    f = np.asarray(instance['fixed_cost'], dtype=float)
    s = np.asarray(instance['capacity'], dtype=float)
    d = np.asarray(instance['demand'], dtype=float)
    c = np.asarray(instance['cost_matrix'], dtype=float)
    m, n = c.shape
    full = (1 << m) - 1
    bits = 1 << np.arange(m)
    masks = np.arange(full + 1)
    selected = (masks[:, None] & bits) != 0
    amounts, quantity_den = _dyadic([*s, *d])
    capacity_units, demand_units = amounts[:m], amounts[m:]
    feasible = _capacity_table(capacity_units, sum(demand_units))
    prices, price_den = _dyadic([*f, *c.ravel()])
    # Keep LP prices on a fine grid even when all input costs are integers.
    finer = max(1, (1 << 52) // price_den)
    price_den *= finer
    prices = [v * finer for v in prices]
    fixed_units = [v * quantity_den for v in prices[:m]]
    cost_units = [prices[m+i*n:m+(i+1)*n] for i in range(m)]
    offsets = [min(row[j] for row in cost_units) for j in range(n)]
    cost_units = [[v - off for v, off in zip(row, offsets)] for row in cost_units]
    offset_value = sum(off * amount for off, amount in zip(offsets, demand_units))
    objective_den = quantity_den * price_den
    if not feasible[full]:
        raise ValueError('Total capacity is below total demand; no feasible incumbent exists.')

    customer = kron(np.ones((1, m)), eye(n), format='csr')
    facility = kron(eye(m), np.ones((1, n)), format='csr')
    equal = hstack([csr_matrix((n, m)), customer], format='csr')
    less = vstack([
        hstack([-diags(s), facility]),
        hstack([-kron(eye(m), d[:, None]), eye(m * n)]),
    ], format='csr')
    # Removing a customer-specific cost constant preserves every feasible
    # flow comparison by the demand equalities. Exact arithmetic below adds
    # the same constant back only when reporting an objective or bound.
    obj = np.r_[f, [v / price_den for row in cost_units for v in row]]
    zeros = np.zeros(less.shape[0])
    var_low = np.zeros(m + m * n)
    var_high = np.r_[np.ones(m), np.full(m * n, inf)]

    # A MIP solution is only an incumbent proposal, never the certificate.
    start = milp(
        obj, integrality=np.r_[np.ones(m), np.zeros(m * n)],
        bounds=Bounds(var_low, var_high),
        constraints=LinearConstraint(vstack([equal, less], format='csr'),
                                     np.r_[d, np.full(len(zeros), -inf)],
                                     np.r_[d, zeros]),
        options={'mip_rel_gap': 0.0},
    )
    upper = inf
    best_value, best_mask, best_flow = None, None, None
    tested = {}

    def incumbent(mask):
        nonlocal upper, best_value, best_mask, best_flow
        if not feasible[mask]:
            return
        if mask not in tested:
            ids = np.flatnonzero(selected[mask]).tolist()
            cost, exact_flow = _transport(ids, capacity_units, demand_units, cost_units)
            value = cost + sum(fixed_units[i] for i in ids)
            tested[mask] = value
            if best_value is None or value < best_value:
                flow = np.zeros_like(c)
                flow[ids] = [[v / quantity_den for v in row] for row in exact_flow]
                upper, best_value, best_mask, best_flow = (value + offset_value) / objective_den, value, mask, flow
        return tested[mask]

    def report_lower(value):
        if value == inf:
            return inf
        value += offset_value
        result = value / objective_den
        num, den = result.as_integer_ratio()
        return nextafter(result, -inf) if num * objective_den > value * den else result

    if start.x is not None:
        incumbent(int(bits[start.x[:m] > 0.5].sum()))
    if best_flow is None:
        incumbent(full)

    def bound(prices_lp, closed, opened):
        # Any demand-price vector gives a valid bound. Quantize the LP's
        # proposed prices to the same integer grid as the transport costs.
        u = []
        for v in prices_lp:
            num, den = float(v).as_integer_ratio()
            u.append((num * price_den + den // 2) // den)
        scores = []
        for i, row in enumerate(cost_units):
            reduced = sorted((v - p, d) for v, p, d in zip(row, u, demand_units))
            # Minimize local reduced cost exactly: fill negative arcs in
            # increasing price order, subject to each demand and site capacity.
            left, score = capacity_units[i], fixed_units[i]
            for price, available in reduced:
                if price >= 0 or not left:
                    break
                sent = min(left, available)
                score += sent * price
                left -= sent
            scores.append(score)
        patterns = [(opened, sum(score for i, score in enumerate(scores) if opened & (1 << i)))]
        free = full ^ (closed | opened)
        while free:
            bit = free & -free
            free ^= bit
            score = scores[bit.bit_length() - 1]
            patterns += [(mask | bit, value + score) for mask, value in patterns]
        return (sum(p * d for p, d in zip(u, demand_units))
                + min(value for mask, value in patterns if feasible[mask]))

    nodes = []

    def visit(closed, opened):
        k = len(nodes)
        nodes.append(None)
        if not feasible[full ^ closed]:
            nodes[k] = (closed, opened, inf, None)
            return k
        free = np.flatnonzero((bits & (closed | opened)) == 0)
        if not len(free):
            value = incumbent(opened)
            lower = value
            nodes[k] = (closed, opened, lower, None)
            return k
        low = var_low.copy()
        high = var_high.copy()
        low[:m] = (bits & opened) != 0
        high[:m] = (bits & closed) == 0
        result = linprog(obj, A_ub=less, b_ub=zeros, A_eq=equal, b_eq=d,
                         bounds=np.column_stack([low, high]),
                         method='highs', options=_LP_OPTIONS)
        if not result.success:
            raise RuntimeError(result.message)
        lower = bound(result.eqlin.marginals, closed, opened)
        y = result.x[:m]
        if lower < best_value and np.max(np.abs(y - (y > 0.5))) < 1e-7:
            incumbent(int(bits[y > 0.5].sum()))
        children = None
        if lower < best_value:
            i = int(free[np.argmax(np.minimum(y[free], 1 - y[free]))])
            bit = int(bits[i])
            left, right = (closed | bit, opened), (closed, opened | bit)
            if y[i] > 0.5:
                left, right = right, left
            children = (visit(*left), visit(*right))
        nodes[k] = (closed, opened, lower, children)
        return k

    root = visit(0, 0)
    leaves = {}
    parents = {}
    ready = []

    def enqueue(k):
        if k in parents:
            p = parents[k]
            a, b = nodes[p][3]
            if a in leaves and b in leaves:
                heappush(ready, (-min(leaves[a], leaves[b]), p))

    def collect(k):
        _, _, lb, kids = nodes[k]
        if report_lower(lb) >= upper - _TOL:
            leaves[k] = lb
            enqueue(k)
        else:
            if kids is None:
                # Report every mathematically sound representable bound even
                # when the preferred internal margin is not representable.
                if report_lower(lb) < upper - 1e-5:
                    raise ArithmeticError('The float output cannot express the requested certificate gap.')
                leaves[k] = lb
                enqueue(k)
                return
            a, b = kids
            parents[a] = parents[b] = k
            collect(a)
            collect(b)

    collect(root)
    # A sibling union can use the weaker of its two child bounds: the minimum
    # over the two child relaxations is a disjunctive relaxation of the parent.
    # Merge only as far as needed after taking all single-witness prunings.
    while len(leaves) > 80:
        neg, p = heappop(ready)
        for k in nodes[p][3]:
            del leaves[k]
        leaves[p] = -neg
        enqueue(p)

    def indices(mask):
        return [i for i in range(m) if mask & (1 << i)]

    return {'open_y': selected[best_mask].astype(int).tolist(),
            'flows_X': best_flow, 'upper_bound': upper,
            'proof_leaves': [dict(fixed_closed=indices(nodes[k][0]),
                                  fixed_open=indices(nodes[k][1]), lower_bound=report_lower(lb))
                             for k, lb in leaves.items()]}
