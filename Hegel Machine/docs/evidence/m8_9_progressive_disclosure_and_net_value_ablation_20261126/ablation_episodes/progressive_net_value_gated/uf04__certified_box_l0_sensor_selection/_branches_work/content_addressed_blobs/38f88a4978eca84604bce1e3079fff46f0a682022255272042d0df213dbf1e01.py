"""Best-bound search for thresholded, boxed sparse ridge regression."""
import heapq
import numpy as np
from scipy.linalg import eigvalsh

_TOL = 1e-9
_GAP = 1e-5
_LIMIT = 600


class _Search:
    def __init__(self, instance):
        self.X = np.asarray(instance['X'], dtype=float)
        self.y = np.asarray(instance['y'], dtype=float)
        self.ridge = float(instance['ridge'])
        self.m = float(instance['box'])
        self.lam = float(instance['l0_penalty'])
        self.t = min(_TOL, self.m)
        self.n, self.p = self.X.shape
        self.G = self.X.T @ self.X + self.ridge * np.eye(self.p)
        self.abs_gram = np.abs(self.X).T @ np.abs(self.X) + self.ridge*np.eye(self.p)
        self.c = self.X.T @ self.y
        self.y2 = 0.5 * (self.y @ self.y)
        # Allow for cancellation when evaluating the supporting quadratic.
        self.err = 16 * np.finfo(float).eps * (self.n + self.p + 4) * (
            1 + self.y2 + self.m * np.sum(np.abs(self.c)) +
            self.m**2 * (np.linalg.norm(self.X, 'fro')**2 * self.p + self.ridge*self.p)
            + self.p * self.lam)
        self.best = np.inf
        self.w = np.zeros(self.p)
        self.nodes = 0
        self.gap = _GAP
        self.full_diag = self._minorant(np.arange(self.p)) if self.p else self.w

    def _minorant(self, ids):
        G = self.G[np.ix_(ids, ids)]
        h = np.diag(G).copy()
        q = np.sqrt(h)
        gram = G / q[:, None] / q
        off = np.abs(gram - np.eye(len(ids)))
        # Spectral and row-sum estimates propose curvature; the Cholesky
        # reconstruction and a rounding margin certify it independently.
        g = np.maximum(0., 1. - off.sum(axis=1))
        e = max(0., float(eigvalsh(gram, subset_by_index=[0, 0])[0]))
        scale = self.abs_gram[np.ix_(ids, ids)].sum(axis=1).max()
        pad = 64*np.finfo(float).eps*(self.n+self.p+4)*scale
        cushion = pad/h
        eye = np.eye(len(ids))
        for estimate in (0.5*g + 0.5*e, e):
            d = h * np.maximum(0., estimate-4*cushion)
            if not np.any(d):
                continue
            shifted = G-np.diag(d)-pad*eye
            try:
                L = np.linalg.cholesky(shifted)
            except np.linalg.LinAlgError:
                continue
            if np.abs(shifted-L@L.T).sum(axis=1).max() <= pad/2:
                return d
        return np.zeros(len(ids))

    def _accept(self, w):
        r = self.X @ w - self.y
        value = 0.5*(r@r + self.ridge*(w@w)) + self.lam*np.count_nonzero(np.abs(w) > _TOL)
        if value < self.best:
            self.best, self.w = float(value), w.copy()

    def _polish(self, support, start):
        bounds = np.where(support, self.m, self.t)
        g = self.G
        w = np.clip(start, -bounds, bounds)
        for _ in range(100):
            old = w.copy()
            for j in range(self.p):
                w[j] = np.clip(w[j] + (self.c[j] - g[j] @ w)/g[j,j], -bounds[j], bounds[j])
            if np.max(np.abs(w-old)) <= 2e-12:
                break
        self._accept(w)

    def _local(self, start):
        w = start.copy()
        h = self.G.diagonal()
        for _ in range(75):
            old = w.copy()
            for j in range(self.p):
                a = abs(w[j]*h[j] + self.c[j] - self.G[j]@w)
                u = min(a/h[j], self.m)
                v = min(a/h[j], self.t)
                b = u if (u-v)*(a-0.5*h[j]*(u+v)) > self.lam else v
                w[j] = np.copysign(b, w[j]*h[j] + self.c[j] - self.G[j]@w)
            if np.max(np.abs(w-old)) <= 1e-10:
                break
        self._accept(w)
        self._polish(np.abs(w)>self.t, w)

    def _relax(self, state, start):
        free = state == 0
        on = state == 1
        off = state == -1
        if not np.any(free):
            from scipy.optimize import lsq_linear
            radius = np.where(on, self.m, self.t)
            A = np.vstack((self.X, np.sqrt(self.ridge)*np.eye(self.p)))
            fit = lsq_linear(A*radius, np.r_[self.y, np.zeros(self.p)],
                             bounds=(-1., 1.), method='bvls', tol=1e-14)
            w = np.clip(fit.x, -1., 1.)*radius
            self._accept(w)
            r = self.X@w - self.y
            grad = self.X.T@r + self.ridge*w
            f = .5*(r@r + self.ridge*(w@w)) + self.lam*np.count_nonzero(on)
            return float(f - np.abs(grad)@radius - grad@w - self.err), -1, w
        ids = np.flatnonzero(~off)
        # Keep a globally valid split; a face-restricted split may sharpen it
        # when inactive intervals have already been fixed (handled below).
        d = self.full_diag.copy()
        if ids.size and ids.size != self.p:
            face = self._minorant(ids)
            # Difference between the face tangent and the full tangent is
            # controlled by a bounded cross term involving inactive variables.
            if face[free[ids]].sum() > d[free].sum()*1.05:
                d[ids] = face
                d[off] = 0.
                face_split = True
            else:
                face_split = False
        else:
            face_split = False
        H = self.G - np.diag(d)
        diag = H.diagonal()
        a = np.maximum(d, np.finfo(float).tiny)
        # Contact of the outer quadratic with the line from the inactive kink.
        b = np.minimum(self.m, np.hypot(self.t, np.sqrt(2*self.lam/a)))
        slope = self.lam/np.maximum(b-self.t, np.finfo(float).tiny) + 0.5*d*(b+self.t)
        bound = np.where(off, self.t, self.m)
        w = np.clip(start, -bound, bound)
        grad = H @ w - self.c
        for sweep in range(260):
            delta = 0.
            for j in range(self.p):
                f = diag[j]*w[j]-grad[j]
                af = abs(f)
                h = self.G[j,j]
                if not free[j]:
                    z = min(af/h, bound[j])
                elif af <= h*self.t:
                    z = af/h
                elif af <= slope[j] + diag[j]*self.t:
                    z = self.t
                elif af < slope[j] + diag[j]*b[j]:
                    z = (af-slope[j])/diag[j]
                else:
                    z = min(self.m, max(b[j], af/h))
                step = np.copysign(z, f)-w[j]
                if step:
                    w[j] += step
                    grad += step*H[:,j]
                    delta = max(delta, abs(step))
            if delta <= 2e-11*(1+self.m):
                break
        grad = H@w-self.c
        g = np.abs(grad)
        # Exact minima of the separated quadratics plus the binary charge.
        v = np.divide(g, a, out=np.full_like(g,self.t), where=g < a*self.t)
        u = np.divide(g, a, out=np.full_like(g,self.m), where=g < a*self.m)
        lo0 = v*(0.5*d*v-g)
        lo1 = u*(0.5*d*u-g)+self.lam
        terms = np.where(off, lo0, np.where(on, lo1, np.minimum(lo0, lo1)))
        lower = self.y2 - 0.5*(w@(H@w)) + terms.sum() - self.err
        if face_split:
            # H on the face and on the inactive block is PSD. Only their
            # cross term can be negative. Bound it for all feasible x-w.
            Hcross = np.abs(H[np.ix_(off, ~off)])
            lower -= (self.t+np.abs(w[off])) @ Hcross @ (self.m+np.abs(w[~off]))
        self._accept(w)
        if self.nodes <= 30 or not np.any(free):
            for factor in (0.25, 0.8):
                support = on | (free & (np.abs(w) >= np.maximum(2*self.t, factor*b)))
                self._polish(support, w)
        # Branch on the largest discrepancy between original and relaxed cost.
        z = np.abs(w)
        envelope = np.where(z <= self.t, 0.5*d*z*z,
                      np.where(z < b, 0.5*d*self.t**2 + slope*(z-self.t), 0.5*d*z*z+self.lam))
        off_loss = np.maximum(0., z-self.t)*np.maximum(g, self.G.diagonal()*np.maximum(0., z-self.t)/2)
        on_loss = np.maximum(0., .5*d*z*z+self.lam-envelope)
        score = np.minimum(on_loss, off_loss) + 0.15*on_loss*np.minimum(1., z/np.maximum(b, 1e-300))
        branch = int(np.argmax(np.where(free, score, -np.inf))) if np.any(free) else -1
        return float(lower), branch, w

    def solve(self):
        self._accept(self.w)
        if not self.p:
            return self.result(max(0., self.best-self.err))
        if self.m <= _TOL:
            self._polish(np.ones(self.p, dtype=bool), self.w)
            grad = self.G@self.w-self.c
            self.nodes = 1
            return self.result(max(0., self.best + np.sum(-self.m*np.abs(grad)-grad*self.w)-self.err))
        start = np.clip(np.linalg.solve(self.G, self.c), -self.m, self.m)
        self._local(start)
        self._local(np.zeros(self.p))
        queue = []
        closed = np.inf
        def add(state, w, parent):
            nonlocal closed
            self.nodes += 1
            lb, j, w = self._relax(state, w)
            lb = max(parent, lb)
            if lb >= self.best-self.gap:
                closed = min(closed, lb)
            elif j >= 0:
                heapq.heappush(queue, (lb, self.nodes, j, state, w))
            else:
                closed = min(closed, lb)
        add(np.zeros(self.p, dtype=np.int8), start, 0.)
        while queue and self.best-min(queue[0][0],closed) > self.gap and self.nodes+2 <= _LIMIT:
            lb, _, j, state, w = heapq.heappop(queue)
            if lb >= self.best-self.gap:
                closed = min(closed, lb)
                continue
            for value in (-1, 1):
                child = state.copy(); child[j] = value
                add(child, w, lb)
        return self.result(min(closed, queue[0][0] if queue else np.inf))

    def result(self, lower):
        return {'weights': self.w.tolist(), 'objective': self.best,
                'certified_lower_bound': float(min(self.best, lower)),
                'nodes_explored': self.nodes}


def _correlation_groups(search):
    X = search.X
    nrm = np.linalg.norm(X, axis=0)
    nrm[nrm == 0] = 1.
    Z = X / nrm
    C = np.abs(Z.T @ Z)
    groups = [[j] for j in range(search.p)]
    while True:
        value, pair = .88, None
        for i, a in enumerate(groups):
            for j in range(i):
                b = groups[j]
                if len(a)+len(b) <= 4:
                    v = C[np.ix_(a,b)].min()
                    if v > value:
                        value, pair = v, (i,j)
        if pair is None:
            return [np.asarray(g) for g in groups]
        i,j = pair
        groups[j] += groups.pop(i)


class _GroupOracle:
    """Exact small box-QP disjunction; each row is a candidate active face."""
    def __init__(self, B, box, cost):
        from itertools import product
        k = len(B)
        offsets, maps, masks = [], [], []
        for face in product(range(4), repeat=k):
            # 0 = inactive, 1 = free, 2/3 = at a signed box end.
            f = np.asarray(face)
            t = np.flatnonzero(f == 1)
            b = box*((f == 2).astype(float)-(f == 3))
            inverse = np.zeros((k,k))
            if len(t):
                if not np.any(B):
                    continue
                inverse[np.ix_(t,t)] = np.linalg.inv(B[np.ix_(t,t)])
            offsets.append(b-inverse @ B @ b)
            maps.append(inverse)
            masks.append(sum(1 << j for j in range(k) if face[j]))
        self.b = np.array(offsets)
        self.A = np.array(maps)
        self.masks = np.array(masks)
        self.penalty = np.array([cost*int(i).bit_count() for i in masks])
        self.counts = np.array([int(i).bit_count() for i in range(1 << k)])
        self.active = (self.masks[:,None] & (1 << np.arange(k))) != 0
        self.B, self.box, self.k = B, box, k

    def oracle(self, a, allowed):
        x = np.clip(self.b-self.A@a, -self.box, self.box)
        grad = x@self.B + a
        q = np.einsum('ij,ij->i',x,grad)*.5+.5*(x@a)+self.penalty
        accept = allowed[self.masks]
        j = np.argmin(np.where(accept,q,np.inf))
        # A feasible face point gives an upper value; its supporting plane
        # over that indicator's box gives a lower value regardless of solves.
        error = np.sum(np.where(self.active, np.abs(grad)*self.box + grad*x, 0.), axis=1)
        mask_bounds = np.full(len(allowed), -np.inf)
        np.maximum.at(mask_bounds, self.masks, q-error)
        return x[j], float(q[j]), float(mask_bounds[allowed].min()), self.masks[j]


class _GroupSearch:
    def __init__(self, s, groups, zero_error):
        self.s, self.groups, self.zero_error = s, groups, zero_error
        p = s.p
        U = np.zeros((p,p))
        vals = np.zeros(p)
        for g in groups:
            v,Q = np.linalg.eigh(s.G[np.ix_(g,g)])
            vals[g] = v
            U[np.ix_(g,g)] = Q
        pad = 128*np.finfo(float).eps*(s.n+p+4)*s.abs_gram.sum(axis=1).max()
        W = U / np.sqrt(vals)
        C = W.T @ (s.G-pad*np.eye(p)) @ W
        C = (C+C.T)*.5
        lo = np.linalg.eigvalsh(C)[0]
        d = np.full(p, max(0., .9*lo))
        if lo > 0:
            R = np.linalg.inv(C-np.diag(d))
            for j in np.argsort(np.abs(C-np.diag(C.diagonal())).sum(axis=1)):
                amt = min(.99-d[j], .91/R[j,j])
                if amt > 0:
                    d[j] += amt
                    R += (amt/(1-amt*R[j,j]))*np.outer(R[:,j],R[j,:])
        B = (U*(d*vals)) @ U.T
        B = (B+B.T)*.5
        # Certify the split in original coordinates, not in the proposal basis.
        Gminus = s.G-B-pad*np.eye(p)
        try:
            L = np.linalg.cholesky(Gminus)
            ok = np.abs(Gminus-L@L.T).sum(axis=1).max() <= pad/2
        except np.linalg.LinAlgError:
            ok = False
        if not ok or np.any(d <= 0):
            B[:] = 0.
            ok = False
        self.H = s.G-B
        self.oracles = [_GroupOracle(B[np.ix_(g,g)], s.m, s.lam) if np.all(d[g]>0) and ok
                        else _GroupOracle(np.zeros((len(g),len(g))), s.m, s.lam) for g in groups]

    def _relax(self, allowed, w, parent, rounds):
        from scipy.optimize import minimize
        s, H = self.s, self.H
        best_lb = parent
        atoms, costs = [], []
        alpha = np.empty(0)
        branch_info = None
        for it in range(rounds):
            h = H@w
            a = h-s.c
            atom = np.empty(s.p)
            lb, value, masks = 0., 0., []
            for g, oracle, choices in zip(self.groups, self.oracles, allowed):
                x, q, l, mask = oracle.oracle(a[g], choices)
                atom[g] = x
                value += q
                lb += l
                masks.append(mask)
            val = s.y2-.5*(w@h)+lb-self.zero_error-s.err
            if val > best_lb or branch_info is None:
                best_lb = max(best_lb, val)
                branch_info = (w.copy(), masks)
            if best_lb >= s.best-8e-6:
                break
            cost = value-h@atom
            if it == 0 or (it == rounds-1 and s.nodes < 50):
                s._polish(np.abs(atom)>s.t, atom)
                if best_lb >= s.best-8e-6:
                    break
            if not any(np.array_equal(atom,v) for v in atoms):
                atoms.append(atom)
                costs.append(cost)
                alpha = np.r_[alpha, 0.] if len(alpha) else np.ones(1)
            if len(atoms) == 1:
                w = atom
                continue
            A = np.asarray(atoms)
            Q = A @ H @ A.T
            c = np.asarray(costs)
            scale = max(1., abs(c).max(), abs(Q).max())
            Q, c = Q/scale, c/scale
            res = minimize(lambda z: (.5*(z@Q@z)+c@z, Q@z+c), alpha, jac=True,
                           method='SLSQP', bounds=[(0.,1.)]*len(A),
                           constraints={'type':'eq','fun':lambda z:z.sum()-1.,
                                        'jac':lambda z:np.ones(len(z))},
                           options={'ftol': 2e-14, 'maxiter': 60})
            alpha = res.x
            nxt = alpha @ A
            if np.max(np.abs(nxt-w)) < 2e-11:
                break
            w = nxt
        center, masks = branch_info
        if best_lb >= s.best-8e-6:
            return best_lb, center, None
        choices = []
        for i,(g,oracle,active,m) in enumerate(zip(self.groups,self.oracles,allowed,masks)):
            if active.sum() <= 1:
                continue
            counts = np.unique(oracle.counts[active])
            atoms_masks = np.asarray([np.sum((np.abs(v[g])>s.t)*(1<<np.arange(len(g)))) for v in atoms],dtype=int)
            mass = np.bincount(oracle.counts[atoms_masks],weights=alpha,minlength=len(g)+1) if len(alpha) else np.zeros(len(g)+1)
            if len(counts)>1:
                # First resolve fractional cardinalities, leaving interchangeable
                # members of a given cardinality in one disjunctive node.
                cum = np.cumsum(mass)[counts[:-1]]
                k = int(counts[np.argmax(np.minimum(cum,1-cum))]) if np.ptp(cum)>1e-10 or np.max(np.minimum(cum,1-cum))>1e-10 else min(int(oracle.counts[m]),int(counts[-2]))
                take = oracle.counts <= k
                priority = 1. + max(0.,float(np.max(np.minimum(cum,1-cum))))*len(g)
            else:
                bits = (np.arange(len(active))[:,None] & (1<<np.arange(len(g)))) != 0
                feasible = bits[active]
                f = alpha @ bits[atoms_masks] if len(alpha) else feasible.mean(axis=0)
                valid = feasible.any(axis=0)&(~feasible.all(axis=0))
                balance = np.where(valid,np.minimum(f,1-f),-1.)
                j = np.argmax(balance)
                take = bits[:,j]
                priority = max(0.,float(balance[j]))
            choices.append((priority, i, take))
        if not choices:
            return best_lb, center, None
        _,i,take = max(choices,key=lambda v:v[0])
        return best_lb, center, (i,take)

    def solve(self):
        s = self.s
        s._accept(s.w)
        start = np.clip(np.linalg.solve(s.G,s.c),-s.m,s.m)
        s._local(start)
        s._local(np.zeros(s.p))
        closed = np.inf
        queue = []
        def add(allowed,w,parent):
            nonlocal closed
            s.nodes += 1
            lb,w,split = self._relax(allowed,w,parent,20)
            if split is None or lb >= s.best-8e-6:
                closed = min(closed,lb)
            else:
                heapq.heappush(queue,(lb,s.nodes,allowed,w,split))
        add(tuple(np.ones(1<<len(g),dtype=bool) for g in self.groups),start,0.)
        while queue and s.best-min(queue[0][0],closed)>_GAP and s.nodes+2<=_LIMIT:
            lb,_,allowed,w,(j,take) = heapq.heappop(queue)
            if lb>=s.best-8e-6:
                closed=min(closed,lb)
                continue
            for part in (take,~take):
                sub = list(allowed)
                sub[j] = sub[j]&part
                add(tuple(sub),w,lb)
        return s.result(min(closed,queue[0][0] if queue else np.inf))


def _exchangeable_root(s):
    """Certify a fully exchangeable design, including sign flips, at the root."""
    x = s.X[:, 0]
    same = np.all(s.X == x[:, None], axis=0)
    flipped = np.all(s.X == -x[:, None], axis=0)
    if not np.all(same | flipped):
        return None
    signs = np.where(same, 1., -1.)
    h = float(x @ x)
    c = float(x @ s.y)
    a = abs(c)
    lower = np.inf
    # At fixed cardinality, permutation symmetry and convexity allow all
    # coordinates in each class to share one coefficient. The on-class box
    # includes zero: this only overcharges points in the inactive interval.
    for k in range(s.p+1):
        q = s.p-k
        if a <= (s.ridge+s.p*h)*s.t:
            v = u = a/(s.ridge+s.p*h)
        else:
            v = s.t
            u = min(s.m, max(0., (a-h*q*v)/(s.ridge+h*k)))
        w = np.copysign(np.r_[np.full(k,u),np.full(q,v)],c)
        radius = np.r_[np.full(k,s.m),np.full(q,s.t)]
        candidate = signs*w
        s._accept(candidate)
        r = s.X @ candidate - s.y
        grad = signs*(s.X.T @ r) + s.ridge*w
        f = .5*(r@r+s.ridge*(w@w))+s.lam*k
        # The minimum of the supporting plane over this cardinality's
        # representative box is a bound even if the stationary point rounds.
        lb = f - (np.abs(grad)@radius + grad@w) - s.err
        lower = min(lower,lb)
    s.nodes = 1
    return s.result(max(0.,lower))


class _CardinalitySearch:
    """Branch on total cardinality, then on indicators within that cardinality."""
    def __init__(self, search):
        self.s = search

    def oracle(self, dual, k, state):
        s = self.s
        # An on indicator pays the charge across the whole box. Including the
        # inactive interval in that box does not change the global minimum.
        a = s.X.T @ dual
        x0 = -np.clip(a/s.ridge, -s.t, s.t)
        x1 = -np.clip(a/s.ridge, -s.m, s.m)
        q0 = x0*(a + .5*s.ridge*x0)
        q1 = x1*(a + .5*s.ridge*x1) + s.lam
        on = state == 1
        free = np.flatnonzero(state == 0)
        count = k - np.count_nonzero(on)
        order = np.argsort((q1-q0)[free], kind='stable')
        on[free[order[:count]]] = True
        w = np.where(on, x1, x0)
        lb = -dual @ (s.y + .5*dual) + np.where(on,q1,q0).sum() - s.err
        return w, on, float(lb)

    def relax(self, k, state, center, parent):
        from scipy.optimize import minimize
        s = self.s
        free = np.flatnonzero(state == 0)
        needed = k - np.count_nonzero(state == 1)
        if needed == 0 or needed == len(free):
            state = state.copy()
            state[free] = 1 if needed else -1
            s._polish(state == 1, center)
            # BVLS supplies a high-accuracy center for a terminal convex QP.
            from scipy.optimize import lsq_linear
            radius = np.where(state == 1, s.m, s.t)
            A = np.vstack([s.X, np.sqrt(s.ridge)*np.eye(s.p)])
            b = np.r_[s.y, np.zeros(s.p)]
            fit = lsq_linear(A*radius, b, bounds=(-1.,1.), method='bvls', tol=1e-14)
            w = np.clip(fit.x,-1.,1.)*radius
            s._accept(w)
            _,_,lb = self.oracle(s.X@w-s.y, k, state)
            return max(parent,lb), -1, w
        atoms, masks, costs = [], [], []
        alpha = np.empty(0)
        lb = parent
        w = center
        for it in range(40):
            atom,on,val = self.oracle(s.X@w-s.y, k, state)
            lb = max(lb,val)
            if lb >= s.best-8e-6:
                break
            if it == 0 or it % 10 == 9:
                s._polish(on, atom)
                if lb >= s.best-8e-6:
                    break
            if any(np.array_equal(atom,old) for old in atoms):
                break
            atoms.append(atom)
            masks.append(on)
            costs.append(.5*s.ridge*(atom@atom))
            alpha = np.r_[alpha,0.] if len(alpha) else np.ones(1)
            A = np.asarray(atoms)
            V = A @ s.X.T
            Q = V @ V.T
            c = np.asarray(costs)-V@s.y
            scale = max(1., abs(Q).max(), abs(c).max())
            Q, c = Q/scale, c/scale
            if len(alpha)>1:
                info = minimize(lambda z: (.5*(z@Q@z)+c@z, Q@z+c),alpha,jac=True,
                                method='SLSQP',bounds=[(0.,1.)]*len(alpha),
                                constraints={'type':'eq','fun':lambda z:z.sum()-1.,
                                             'jac':lambda z:np.ones(len(z))},
                                options={'ftol':1e-14,'maxiter':60})
                alpha = info.x
            nxt = alpha @ A
            if np.max(np.abs(s.X@(nxt-w))) < 1e-12:
                break
            w = nxt
        if lb >= s.best-8e-6:
            return lb, -1, w
        mass = alpha @ np.asarray(masks) if len(alpha) else on.astype(float)
        branch = free[np.argmax(np.minimum(mass[free],1-mass[free]))]
        return lb, int(branch), w

    def solve(self):
        s = self.s
        s._accept(s.w)
        start = np.clip(np.linalg.solve(s.G,s.c),-s.m,s.m)
        s._local(start)
        s._local(s.w)
        queue = []
        closed = np.inf
        def add(k,state,w,parent):
            nonlocal closed
            s.nodes += 1
            lb,j,w = self.relax(k,state,w,parent)
            if j < 0 or lb >= s.best-8e-6:
                closed = min(closed,lb)
            else:
                heapq.heappush(queue,(lb,s.nodes,k,state,j,w))
        s.nodes = 1
        x = s.X[:,0]
        sign = np.where(x@s.X >= 0.,1.,-1.)
        h = float(x@x)
        c = float(x@s.y)
        counts = sorted(range(s.p+1),key=lambda k:abs(k-np.count_nonzero(abs(s.w)>s.t)))
        for k in counts:
            w = sign*np.copysign(min(s.m,abs(c)/(h*max(1,k)+s.ridge)),c)*(k/s.p)
            add(k,np.zeros(s.p,dtype=np.int8),w,k*s.lam-s.err)
        while queue and s.best-min(closed,queue[0][0])>_GAP and s.nodes+2<=_LIMIT:
            lb,_,k,state,j,w = heapq.heappop(queue)
            if lb >= s.best-8e-6:
                closed = min(closed,lb)
                continue
            for v in (-1,1):
                child = state.copy()
                child[j] = v
                add(k,child,w,lb)
        return s.result(min(closed,queue[0][0] if queue else np.inf))


def _independent_blocks(s):
    """Solve small components separately only when their row sets are disjoint."""
    incidence = s.X != 0.
    remaining = np.ones(s.p, dtype=bool)
    blocks = []
    while np.any(remaining):
        cols = np.zeros(s.p, dtype=bool)
        cols[np.flatnonzero(remaining)[0]] = True
        while True:
            rows = np.any(incidence[:, cols], axis=1)
            more = np.any(incidence[rows], axis=0) & ~cols
            if not np.any(more):
                break
            cols |= more
        if np.count_nonzero(cols) > 4:
            return None
        blocks.append((rows, cols))
        remaining[cols] = False
    if len(blocks) < 2:
        return None
    covered = np.zeros(s.n, dtype=bool)
    w = np.zeros(s.p)
    lower, nodes = 0., 1
    for rows, cols in blocks:
        if not np.any(rows):
            continue  # Zero columns contribute exactly zero at w=0.
        covered |= rows
        part = _Search(dict(X=s.X[rows][:, cols], y=s.y[rows], ridge=s.ridge,
                            box=s.m, l0_penalty=s.lam))
        part.gap = _GAP/(2*len(blocks))
        exact = _exchangeable_root(part) if s.m > _TOL else None
        result = part.solve() if exact is None else exact
        w[cols] = result['weights']
        lower += result['certified_lower_bound']
        nodes += result['nodes_explored']
    unused = s.y[~covered]
    lower += .5*(unused@unused)
    s._accept(w)
    s.nodes = nodes
    return s.result(max(0., lower-s.err))


def solve_certified_box_l0(instance: dict) -> dict:
    search = _Search(instance)
    if search.p and search.m > _TOL:
        independent = _independent_blocks(search)
        if independent is not None:
            return independent
        exact = _exchangeable_root(search)
        if exact is not None:
            return exact
        norm = np.linalg.norm(search.X, axis=0)
        if np.all(norm > 0):
            Z = search.X / norm
            if np.min(np.abs(Z.T@Z)) > .96:
                return _CardinalitySearch(search).solve()
        groups = _correlation_groups(search)
        if len(groups) < search.p:
            # Moving only uncharged coordinates to zero changes the smooth
            # loss by at most t * max gradient L1 on the original box.
            error = search.t*(np.abs(search.c).sum()+search.m*search.abs_gram.sum())
            if error < 1e-6:
                return _GroupSearch(search, groups, error).solve()
    return search.solve()
