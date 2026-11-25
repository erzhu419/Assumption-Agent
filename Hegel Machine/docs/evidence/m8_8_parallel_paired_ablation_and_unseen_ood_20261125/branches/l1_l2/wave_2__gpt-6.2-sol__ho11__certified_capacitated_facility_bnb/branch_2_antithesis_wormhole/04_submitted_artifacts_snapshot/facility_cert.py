"""Branch-and-bound for capacitated facility location with splittable demand."""
import math
from heapq import heappop, heappush

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack


_LP_OPTIONS = {'primal_feasibility_tolerance': 1e-9,
               'dual_feasibility_tolerance': 1e-9}


def _dyadics(values):
    """Represent binary64 data exactly with one power-of-two denominator."""
    ratios = [float(v).as_integer_ratio() for v in values]
    den = max(q for _, q in ratios)
    return [p * (den // q) for p, q in ratios], den


def _down(value, den):
    result = value / den
    p, q = result.as_integer_ratio()
    return math.nextafter(result, -math.inf) if p * den > value * q else result


class _FacilityProblem:
    def __init__(self, instance):
        self.f, self.u, self.d, self.c = (
            np.asarray(instance[k], dtype=float)
            for k in ('fixed_cost', 'capacity', 'demand', 'cost_matrix')
        )
        self.m, self.n = m, n = self.c.shape
        self.bits = 1 << np.arange(m)
        self.all_open = (1 << m) - 1
        weights, self.quantity_den = _dyadics(np.r_[self.u, self.d])
        self.cap, self.dem = weights[:m], weights[m:]
        self.need = need = sum(self.dem)
        costs, self.cost_den = _dyadics(np.r_[self.f, self.c.ravel()])
        self.fixed = costs[:m]
        self.cost = [costs[m+i*n:m+(i+1)*n] for i in range(m)]
        self.scale = self.cost_den*self.quantity_den
        total = [0] * (1 << m)
        feasible = []
        for mask in range(1, 1 << m):
            bit = mask & -mask
            total[mask] = total[mask ^ bit] + self.cap[bit.bit_length() - 1]
            if total[mask] >= need:
                feasible.append(mask)
        if not feasible:
            raise ValueError('Total facility capacity is below total demand')
        self.feasible = np.array(feasible, dtype=int)
        self.is_feasible = set(feasible)

        # HiGHS sees flows in demand-sized units; the exact solver and
        # certificate continue to use the original dyadic data.
        self.lp_exponent = math.frexp(float(self.d.max()))[1] - 2
        lp_u = np.ldexp(self.u, -self.lp_exponent)
        self.lp_d = np.ldexp(self.d, -self.lp_exponent)
        lp_c = np.ldexp(self.c, self.lp_exponent)
        i = np.repeat(np.arange(m), n)
        j = np.tile(np.arange(n), m)
        a = np.arange(m*n)
        x = m + a
        size = m + m*n
        self.eq = coo_matrix((np.ones(m*n), (j, x)), shape=(n, size)).tocsr()
        # Capacity links and the valid inequalities X[i,j] <= d[j] y[i].
        self.ub = coo_matrix(
            (np.r_[-lp_u, np.ones(2*m*n), -self.lp_d[j]],
             (np.r_[np.arange(m), i, m+a, m+a], np.r_[np.arange(m), x, x, i])),
            shape=(size, size)).tocsr()
        self.obj = np.r_[self.f, lp_c.ravel()]
        self.zero = np.zeros(size)
        self.upper = math.inf
        self.best_y = self.best_x = None
        self.prices = []
        self.best_bound = np.full(1 << m, -np.inf)
        self.exact_bound = [None] * (1 << m)
        self.exact_upper = None
        self.tried = set()

    def transport(self, mask):
        """Exact min-cost flow for one capacity-feasible opening vector."""
        if mask not in self.is_feasible or mask in self.tried:
            return
        self.tried.add(mask)
        y = ((mask & self.bits) != 0).astype(int)
        opened = [int(i) for i in np.flatnonzero(y)]
        k, n = len(opened), self.n
        src, sink = k+n, k+n+1
        graph = [[] for _ in range(sink+1)]

        def arc(u, v, cap, cost):
            fwd = [v, cap, cost, len(graph[v])]
            rev = [u, 0, -cost, len(graph[u])]
            graph[u].append(fwd)
            graph[v].append(rev)
            return rev

        for a, i in enumerate(opened):
            arc(src, a, self.cap[i], 0)
        arcs = [[arc(a, k+j, self.need+1, c) for j, c in enumerate(self.cost[i])]
                for a, i in enumerate(opened)]
        for j, d in enumerate(self.dem):
            arc(k+j, sink, d, 0)
        potential = [0]*(sink+1)
        remaining = self.need
        while remaining:
            dist = [None]*(sink+1)
            prev = [None]*(sink+1)
            dist[src] = 0
            queue = [(0, src)]
            while queue:
                dv, u = heappop(queue)
                if dv != dist[u]:
                    continue
                for e, (v, cap, cost, _) in enumerate(graph[u]):
                    if not cap:
                        continue
                    cand = dv + cost + potential[u] - potential[v]
                    if dist[v] is None or cand < dist[v]:
                        dist[v], prev[v] = cand, (u, e)
                        heappush(queue, (cand, v))
            for v, dv in enumerate(dist):
                if dv is not None:
                    potential[v] += dv
            amount = remaining
            path = []
            v = sink
            while v != src:
                u, e = prev[v]
                edge = graph[u][e]
                amount = min(amount, edge[1])
                path.append((u, edge))
                v = u
            for u, edge in path:
                edge[1] -= amount
                graph[edge[0]][edge[3]][1] += amount
            remaining -= amount

        parts = [[edge[1] for edge in row] for row in arcs]
        value = sum(self.fixed[i]*self.quantity_den +
                    sum(c*q for c, q in zip(self.cost[i], row))
                    for i, row in zip(opened, parts))
        if self.exact_upper is None or value < self.exact_upper:
            self.exact_upper = value
            self.upper = value/self.scale
            self.best_y = y
            self.best_x = np.zeros_like(self.c)
            self.best_x[opened] = [[q/self.quantity_den for q in row] for row in parts]
        # The transport row dual is potential[row]-potential[source].
        # Customer prices give a tight knapsack bound on this opening.
        self.add_price([potential[k+j]-potential[src] for j in range(n)], exact=True)
        if self.exact_bound[mask] != value:
            raise RuntimeError('Exact transport primal and dual disagree')

    def seed(self):
        m, n = self.m, self.n
        r = milp(self.obj, integrality=np.r_[np.ones(m), np.zeros(m*n)],
                 bounds=Bounds(0, np.r_[np.ones(m), np.full(m*n, np.inf)]),
                 constraints=LinearConstraint(vstack([self.eq, self.ub]),
                     np.r_[self.lp_d, np.full(len(self.zero), -np.inf)],
                     np.r_[self.lp_d, self.zero]),
                 options={'mip_rel_gap': 1e-9, 'node_limit': 20})
        if r.x is not None:
            self.transport(int(self.bits @ (r.x[:m] > .5)))
        if self.best_y is None:
            self.transport(self.all_open)

    def relaxation(self, opened, closed):
        bounds = list(zip(((opened & self.bits) != 0).astype(int),
                          ((closed & self.bits) == 0).astype(int)))
        r = linprog(self.obj, A_eq=self.eq, b_eq=self.lp_d, A_ub=self.ub,
                    b_ub=self.zero, bounds=bounds + [(0, None)]*(self.m*self.n),
                    method='highs', options=_LP_OPTIONS)
        # LP guidance is optional: capacity feasibility and pruning are exact.
        return r if r.success else None

    def add_price(self, price, *, exact=False):
        """For any customer price p, dualize demand, not opening decisions.

        Each open row can earn at most the continuous knapsack optimum with
        profits p[j]-c[i,j], item bounds d[j], and capacity u[i]. For a cube,
        minimize the residual fixed costs over capacity-feasible openings.
        Solve both knapsacks exactly over dyadics and round down only at output.
        """
        m, n = self.m, self.n
        if exact:
            p = price
        else:
            # Any price is a valid Lagrangian multiplier. Quantize heuristic
            # prices to the same integer grid used by the exact flow duals.
            p = []
            for v in price:
                num, den = float(v).as_integer_ratio()
                p.append((num*self.cost_den + den//2)//den)
        base = sum(d*v for d, v in zip(self.dem, p))
        residual = []
        for i in range(m):
            gain = [v-c for v, c in zip(p, self.cost[i])]
            left = self.cap[i]
            benefit = 0
            for j in sorted(range(n), key=gain.__getitem__, reverse=True):
                if gain[j] <= 0 or left == 0:
                    break
                use = min(left, self.dem[j])
                benefit += use*gain[j]
                left -= use
            residual.append(self.fixed[i]*self.quantity_den - benefit)
        subtotal = [base] * (1 << m)
        bounds = np.full(1 << m, np.inf)
        for mask in range(1, 1 << m):
            bit = mask & -mask
            subtotal[mask] = subtotal[mask ^ bit] + residual[bit.bit_length()-1]
            if mask in self.is_feasible:
                value = subtotal[mask]
                if self.exact_bound[mask] is None or value > self.exact_bound[mask]:
                    self.exact_bound[mask] = value
                bounds[mask] = _down(value, self.scale)
        self.prices.append(bounds)
        np.maximum(self.best_bound, bounds, out=self.best_bound)

    def finish_exact(self):
        """The reported tolerance is not an optimality rule for the incumbent."""
        while True:
            mask = min(self.is_feasible, key=self.exact_bound.__getitem__)
            if self.exact_bound[mask] >= self.exact_upper:
                return
            self.transport(mask)

    def cutoff(self):
        return self.upper - 2e-6

    def direct_bound(self, masks):
        return max(float(b[masks].min()) for b in self.prices) if len(masks) else math.inf

    def branch(self, opened, closed, masks):
        lb = self.direct_bound(masks)
        if lb >= self.cutoff():
            return (opened, closed, masks, lb, ())
        free = np.flatnonzero(((opened | closed) & self.bits) == 0)
        r = None
        if len(masks) == 1:
            self.transport(int(masks[0]))
        else:
            r = self.relaxation(opened, closed)
            if r is not None:
                self.add_price(np.ldexp(r.eqlin.marginals, -self.lp_exponent))
                self.transport(int(self.bits @ (r.x[:self.m] > .5)))
            else:
                mask = min(map(int, masks), key=self.exact_bound.__getitem__)
                self.transport(mask)
        lb = self.direct_bound(masks)
        if lb >= self.cutoff():
            return (opened, closed, masks, lb, ())
        if len(masks) == 1:
            raise RuntimeError('Transport dual did not close the requested gap')
        if r is None:
            counts = np.count_nonzero(masks[:, None] & self.bits[free], axis=0)
            j = free[np.argmax(np.minimum(counts, len(masks)-counts))]
        else:
            y = r.x[free]
            j = free[np.argmax(np.minimum(y, 1-y))]
        bit = int(self.bits[j])
        side = (masks & bit) != 0
        return (opened, closed, masks, lb,
                (self.branch(opened, closed | bit, masks[~side]),
                 self.branch(opened | bit, closed, masks[side])))

    def compact(self, node, disjunctive=False):
        """Bottom-up: prefer one price per cube; allow a finite price bundle
        if needed. Min over openings of their maximum certified bounds is
        valid even when no single price certifies the entire parent cube.
        """
        opened, closed, masks, lb, children = node
        if children:
            leaves = [leaf for child in children for leaf in self.compact(child, disjunctive)]
            if disjunctive:
                lb = min(leaf['lower_bound'] for leaf in leaves)
                if len(masks):
                    lb = max(lb, float(self.best_bound[masks].min()))
            else:
                lb = self.direct_bound(masks)
            if lb < self.cutoff():
                return leaves
        elif disjunctive and len(masks):
            lb = max(lb, float(self.best_bound[masks].min()))
        return [{'fixed_closed': np.flatnonzero(closed & self.bits).tolist(),
                 'fixed_open': np.flatnonzero(opened & self.bits).tolist(),
                 'lower_bound': lb}]


def solve_and_certify_facilities(instance: dict) -> dict:
    """Return an optimal flow and a compact partition certificate (gap <= 1e-5)."""
    problem = _FacilityProblem(instance)
    problem.seed()
    root = problem.branch(0, 0, problem.feasible)
    problem.finish_exact()
    leaves = problem.compact(root)
    if len(leaves) > 80:
        leaves = problem.compact(root, disjunctive=True)
    return {'open_y': problem.best_y.tolist(), 'flows_X': problem.best_x,
            'upper_bound': problem.upper, 'proof_leaves': leaves}
