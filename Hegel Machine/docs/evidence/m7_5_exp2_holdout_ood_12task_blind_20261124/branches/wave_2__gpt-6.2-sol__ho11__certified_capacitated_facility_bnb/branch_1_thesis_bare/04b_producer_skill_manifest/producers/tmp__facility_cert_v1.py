"""Branch-and-bound for capacitated facility location with splittable demand."""
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix


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
        ratios = [float(v).as_integer_ratio() for v in np.r_[self.cap, self.demand]]
        denominator = max(q for _, q in ratios)
        units = [p * (denominator // q) for p, q in ratios]
        total = sum(units[m:])
        capacities = [0] * (1 << m)
        for mask in range(1, 1 << m):
            bit = mask & -mask
            capacities[mask] = capacities[mask ^ bit] + units[bit.bit_length()-1]
        self.enough = [value >= total for value in capacities]
        self.feasible = np.flatnonzero(self.enough)
        self.opens = ((self.feasible[:, None] & self.bits) != 0).astype(float)

    def _lp(self, low, high):
        return linprog(self.obj, A_ub=self.ineq, b_ub=self.zero,
                       A_eq=self.eq, b_eq=self.demand,
                       bounds=np.column_stack((np.r_[low, np.zeros(self.m*self.n)],
                                               np.r_[high, np.full(self.m*self.n, np.inf)])),
                       method='highs')

    def get_incumbent(self):
        m, n = self.m, self.n
        r = milp(self.obj, integrality=np.r_[np.ones(m), np.zeros(m*n)],
                 bounds=Bounds(np.zeros(m+m*n), np.r_[np.ones(m), np.full(m*n, np.inf)]),
                 constraints=[LinearConstraint(self.eq, self.demand, self.demand),
                              LinearConstraint(self.ineq, -np.inf, self.zero)],
                 options={'mip_rel_gap': 0.0})
        if not r.success:
            raise RuntimeError(r.message)
        y = np.rint(r.x[:m]).astype(int)
        sol = self._lp(y, y)
        if not sol.success:
            raise RuntimeError(sol.message)
        return y, sol.x[m:].reshape(m, n), float(self.obj @ sol.x)

    def bound(self, alpha, eligible):
        # Relax customer equalities with multipliers alpha. Conditional on a
        # facility being open, its reduced-cost problem is a fractional knapsack.
        reduced = self.cost - alpha
        order = np.argsort(reduced, axis=1)
        sorted_cost = np.take_along_axis(reduced, order, axis=1)
        quantity = self.demand[order]
        prefix = np.cumsum(quantity, axis=1) - quantity
        quantity = np.minimum(quantity, np.maximum(0, self.cap[:, None] - prefix))
        coefficient = self.fixed + np.sum(np.minimum(sorted_cost, 0) * quantity, axis=1)
        # Keep the necessary integral capacity cover in the relaxation.
        value = alpha @ self.demand + (self.opens[eligible] @ coefficient).min()
        scale = 1 + np.abs(alpha) @ self.demand + np.sum(self.fixed)
        scale += np.maximum(-reduced, 0).max() * np.sum(self.cap)
        return float(value - 1e-12 * scale)

    def certify(self, upper):
        target = upper - 1e-5

        def visit(closed, opened, eligible):
            if not self.enough[self.full ^ closed]:
                return [(closed, opened, math.inf)]
            low = (opened & self.bits) != 0
            high = (closed & self.bits) == 0
            lp = self._lp(low, high)
            if not lp.success:
                raise RuntimeError(lp.message)
            lower = self.bound(lp.eqlin.marginals, eligible)
            if lower >= target:
                return [(closed, opened, lower)]
            free = np.flatnonzero(high & ~low)
            if not len(free):
                raise RuntimeError('Incumbent and certified full-assignment bound disagree')
            fractions = lp.x[free]
            bit = 1 << int(free[np.argmax(np.minimum(fractions, 1-fractions))])
            left = visit(closed | bit, opened, eligible[(self.feasible[eligible] & bit) == 0])
            right = visit(closed, opened | bit, eligible[(self.feasible[eligible] & bit) != 0])
            return left + right

        return visit(0, 0, np.arange(len(self.feasible)))


def solve_and_certify_facilities(instance: dict) -> dict:
    """Return an optimal solution and a partition certified by relaxation bounds."""
    p = _Problem(instance)
    y, flows, upper = p.get_incumbent()
    leaves = [{'fixed_closed': [i for i in range(p.m) if c & (1 << i)],
               'fixed_open': [i for i in range(p.m) if o & (1 << i)],
               'lower_bound': bound} for c, o, bound in p.certify(upper)]
    return {'open_y': y.tolist(), 'flows_X': flows, 'upper_bound': upper,
            'proof_leaves': leaves}
