"""Branch-and-bound for capacitated facility location with splittable demand."""
from fractions import Fraction
from heapq import heappop, heappush
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix


def _dyadic(values):
    ratios = [float(v).as_integer_ratio() for v in values]
    scale = max(q for _, q in ratios)
    return [p * (scale // q) for p, q in ratios], scale


def _round_down(value):
    result = float(value)
    return math.nextafter(result, -math.inf) if result > value else result


class _Problem:
    def __init__(self, instance):
        self.fixed = np.asarray(instance['fixed_cost'], dtype=float)
        self.cap = np.asarray(instance['capacity'], dtype=float)
        self.demand = np.asarray(instance['demand'], dtype=float)
        self.cost = np.asarray(instance['cost_matrix'], dtype=float)
        self.m, self.n = self.cost.shape
        m, n = self.m, self.n
        self.full = (1 << m) - 1
        self.bits = 1 << np.arange(m)
        self.obj = np.r_[self.fixed, self.cost.ravel()]
        i = np.repeat(np.arange(m), n)
        j = np.tile(np.arange(n), m)
        k = np.arange(m * n)
        ones = np.ones(m * n)
        shape = (m + m*n, m + m*n)
        self.ineq = csr_matrix((np.r_[-self.cap, ones, -self.demand[j], ones],
                               (np.r_[np.arange(m), i, m+k, m+k],
                                np.r_[np.arange(m), m+k, i, m+k])), shape=shape)
        self.eq = csr_matrix((ones, (j, m+k)), shape=(n, m + m*n))
        self.zero = np.zeros(shape[0])
        # Binary64 inputs are dyadic rationals. Compare capacities exactly.
        units, self.qty_scale = _dyadic(np.r_[self.cap, self.demand])
        self.cap_int, self.demand_int = units[:m], units[m:]
        total = self.total = sum(self.demand_int)
        costs, self.cost_scale = _dyadic(self.obj)
        self.fixed_int = costs[:m]
        self.cost_int = [costs[m+i*n:m+(i+1)*n] for i in range(m)]
        capacities = [0] * (1 << m)
        for mask in range(1, 1 << m):
            bit = mask & -mask
            capacities[mask] = capacities[mask ^ bit] + units[bit.bit_length()-1]
        self.enough = [value >= total for value in capacities]
        self.feasible = np.flatnonzero(self.enough)
        self.members = [tuple(i for i in range(m) if mask & (1 << i))
                        for mask in range(1 << m)]

    def _lp(self, low, high):
        return linprog(self.obj, A_ub=self.ineq, b_ub=self.zero,
                       A_eq=self.eq, b_eq=self.demand,
                       bounds=np.column_stack((np.r_[low, np.zeros(self.m*self.n)],
                                               np.r_[high, np.full(self.m*self.n, np.inf)])),
                       method='highs')

    def starting_mask(self):
        m, n = self.m, self.n
        r = milp(self.obj, integrality=np.r_[np.ones(m), np.zeros(m*n)],
                 bounds=Bounds(np.zeros(m+m*n), np.r_[np.ones(m), np.full(m*n, np.inf)]),
                 constraints=[LinearConstraint(self.eq, self.demand, self.demand),
                              LinearConstraint(self.ineq, -np.inf, self.zero)])
        if r.x is not None:
            mask = sum(1 << i for i in range(m) if r.x[i] >= 0.5)
            if self.enough[mask]:
                return mask
        return self.full

    def transport(self, mask):
        # Exact successive shortest augmenting paths in the transportation
        # network. Integer units represent the binary64 inputs without error.
        m, n = self.m, self.n
        source, sink = m+n, m+n+1
        graph = [[] for _ in range(sink+1)]

        def edge(u, v, cap, cost):
            forward = [v, len(graph[v]), cap, cost]
            graph[v].append([u, len(graph[u]), 0, -cost])
            graph[u].append(forward)
            return forward

        arcs = {}
        for i in self.members[mask]:
            edge(source, i, self.cap_int[i], 0)
            arcs[i] = [edge(i, m+j, self.demand_int[j], self.cost_int[i][j])
                       for j in range(n)]
        for j, amount in enumerate(self.demand_int):
            edge(m+j, sink, amount, 0)

        potential = [0] * len(graph)
        remaining = self.total
        while remaining:
            dist, prev = [None] * len(graph), [None] * len(graph)
            dist[source] = 0
            queue = [(0, source)]
            while queue:
                d, u = heappop(queue)
                if d != dist[u]:
                    continue
                for idx, (v, _, cap, cost) in enumerate(graph[u]):
                    if not cap:
                        continue
                    nd = d + cost + potential[u] - potential[v]
                    if dist[v] is None or nd < dist[v]:
                        dist[v], prev[v] = nd, (u, idx)
                        heappush(queue, (nd, v))
            for v, d in enumerate(dist):
                if d is not None:
                    potential[v] += d
            path, amount, v = [], remaining, sink
            while v != source:
                u, idx = prev[v]
                arc = graph[u][idx]
                amount = min(amount, arc[2])
                path.append((v, graph[v][arc[1]], arc))
                v = u
            for _, reverse, forward in path:
                forward[2] -= amount
                reverse[2] += amount
            remaining -= amount

        flows = np.zeros((m, n))
        total = sum(self.fixed_int[i] for i in self.members[mask]) * self.qty_scale
        for i, row in arcs.items():
            for j, (_, reverse, _, cost) in enumerate(row):
                amount = graph[m+j][reverse][2]
                flows[i, j] = amount / self.qty_scale
                total += amount * cost
        return Fraction(total, self.qty_scale * self.cost_scale), flows

    def bound(self, alpha, eligible):
        # Drop the customer balance constraints at prices alpha. For a fixed
        # facility the remaining flow problem is a fractional knapsack: fill
        # negative reduced-cost arcs in increasing cost order, up to demand.
        # Minimize the resulting open costs over integral capacity covers.
        prices, scale = _dyadic(alpha)
        common = max(scale, self.cost_scale)
        prices = [v * (common // scale) for v in prices]
        coefficient = []
        for i, row in enumerate(self.cost_int):
            reduced = [c * (common // self.cost_scale) - p for c, p in zip(row, prices)]
            value = self.fixed_int[i] * (common // self.cost_scale) * self.qty_scale
            remaining = self.cap_int[i]
            for j in sorted(range(self.n), key=reduced.__getitem__):
                if not remaining or reduced[j] >= 0:
                    break
                amount = min(remaining, self.demand_int[j])
                value += reduced[j] * amount
                remaining -= amount
            coefficient.append(value)
        base = sum(p*d for p, d in zip(prices, self.demand_int))
        best = min(sum(coefficient[i] for i in self.members[mask]) for mask in eligible)
        return Fraction(base + best, common * self.qty_scale)

    def certify(self):
        if not self.enough[self.full]:
            raise ValueError('Total capacity is less than total demand')
        best_mask = self.starting_mask()
        best, flows = self.transport(best_mask)

        def visit(closed, opened, eligible):
            nonlocal best, best_mask, flows
            if not len(eligible):
                return (closed, opened, math.inf)
            if len(eligible) == 1:
                # The exact fixed-pattern transportation LP bounds this cell:
                # all its other patterns fail the capacity cover test.
                mask = int(eligible[0])
                if mask == best_mask:
                    lower = best
                else:
                    lower, x = self.transport(mask)
                    if lower < best:
                        best, best_mask, flows = lower, mask, x
                return (closed, opened, lower)
            low = (opened & self.bits) != 0
            high = (closed & self.bits) == 0
            lp = self._lp(low, high)
            if lp.success:
                lower = self.bound(lp.eqlin.marginals, eligible)
                if lower >= best:
                    return (closed, opened, lower)
            free = np.flatnonzero(high & ~low)
            if lp.success:
                fractions = lp.x[free]
                i = free[np.argmax(np.minimum(fractions, 1-fractions))]
            else:
                i = free[0]
            bit = 1 << int(i)
            opened_here = (eligible & bit) != 0
            left = visit(closed | bit, opened, eligible[~opened_here])
            right = visit(closed, opened | bit, eligible[opened_here])
            return (closed, opened, min(left[2], right[2]), left, right)

        tree = visit(0, 0, self.feasible)
        return best_mask, flows, best, self.compact(tree)

    @staticmethod
    def compact(tree):
        # A sibling pair covers its parent, which inherits the minimum of
        # their certified bounds. Merge deepest sibling pairs first.
        leaves, parents = {}, []

        def collect(node, index, depth):
            if len(node) == 3:
                leaves[index] = node
            else:
                collect(node[3], 2*index, depth+1)
                collect(node[4], 2*index+1, depth+1)
                heappush(parents, (-depth, index, node[:3]))

        collect(tree, 1, 0)
        while len(leaves) > 80:
            _, index, leaf = heappop(parents)
            del leaves[2*index], leaves[2*index+1]
            leaves[index] = leaf
        return leaves.values()


def solve_and_certify_facilities(instance: dict) -> dict:
    """Return an optimal solution and a certified partition of opening patterns."""
    p = _Problem(instance)
    mask, flows, optimum, proof = p.certify()
    leaves = [{'fixed_closed': list(p.members[c]), 'fixed_open': list(p.members[o]),
               'lower_bound': _round_down(bound)} for c, o, bound in proof]
    y = [int(bool(mask & (1 << i))) for i in range(p.m)]
    return {'open_y': y, 'flows_X': flows, 'upper_bound': float(optimum),
            'proof_leaves': leaves}
