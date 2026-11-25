"""Certified branch-and-bound for boxed L0 ridge regression."""
import heapq
from itertools import count

import numpy as np
from scipy.optimize import lsq_linear


_ZERO = 1e-9
_GAP = 9.8e-6


class _Problem:
    def __init__(self, instance):
        self.x = np.asarray(instance['X'], dtype=float)
        self.y = np.asarray(instance['y'], dtype=float)
        self.ridge = float(instance['ridge'])
        self.box = float(instance['box'])
        self.lam = float(instance['l0_penalty'])
        self.t = min(self.box, _ZERO)
        n, self.p = self.x.shape
        self.a = np.vstack((self.x, np.sqrt(self.ridge) * np.eye(self.p)))
        self.b = np.r_[self.y, np.zeros(self.p)]
        h = self.x.T @ self.x + self.ridge * np.eye(self.p)
        self.c = self.x.T @ self.y
        self.c0 = 0.5 * (self.y @ self.y)
        # Numerical allowance for Gram formation, eigenvalue and bound arithmetic.
        u = np.finfo(float).eps
        xsum = np.abs(self.x).sum(axis=1)
        self.err_h = 16*u*(n+self.p+1)*(np.sum(xsum*xsum)+self.p*self.ridge)
        self.pad = 4*self.err_h*self.box**2 + 16*u*(n+self.p+1)*(self.c0+self.box*(xsum @ np.abs(self.y))+self.p*self.lam)
        eig = np.linalg.eigvalsh(h)
        self.d = max(self.ridge * 0.5, eig[0] - self.err_h)
        self.q = h - self.d*np.eye(self.p)
        self.step = 1 / max(self.d, eig[-1]-self.d)
        self.bridge = min(self.box, self.t + np.sqrt(2*self.lam/self.d))
        if self.bridge > self.t:
            self.slope = 0.5*self.d*(self.bridge+self.t)+self.lam/(self.bridge-self.t)
        else:
            self.slope = 0.0
        self.best = self.c0
        self.w = np.zeros(self.p)
        self.tried = {}
        self.prepare_pairs(h)

    def prepare_pairs(self, h):
        """Split off joint curvature for disjoint pairs of near-duplicate columns."""
        x = self.x
        diag = np.sum(x*x, axis=0)
        j, k = np.triu_indices(self.p, 1)
        den = np.sqrt(diag[j]*diag[k])
        corr = np.divide(np.abs(h[j,k]), den, out=np.zeros_like(den), where=den>0)
        pairs = []
        used = set()
        for z in np.argsort(-corr):
            i, l = int(j[z]), int(k[z])
            if corr[z] < .85:
                break
            if i not in used and l not in used:
                pairs.append((i,l))
                used.update((i,l))
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1,2)
        if not pairs:
            return
        self.single = np.array([i for i in range(self.p) if i not in used], dtype=int)
        rot = np.eye(self.p)
        for i,l in pairs:
            _, rot[np.ix_([i,l],[i,l])] = np.linalg.eigh(h[np.ix_([i,l],[i,l])])
        off = rot.T @ h @ rot
        np.fill_diagonal(off, 0.)
        for i,l in pairs:
            off[i,l] = off[l,i] = 0.
        # In block eigen-coordinates, Gershgorin makes the remainder PSD.
        # Weighted diagonal dominance certifies the PSD remainder. A switch
        # between duplicate columns moves primarily along the low-curvature
        # contrast, so retain more of that direction in the block oracle.
        scale = np.ones(self.p)
        scale[self.pairs[:, 0]] = 8.0  # eigh orders contrast before sum
        q = off + np.diag((np.abs(off) @ scale) / scale + self.err_h)
        self.pair_q = rot @ q @ rot.T
        d = h - self.pair_q
        i,l = self.pairs.T
        self.pair_d = (d[i,i,None], d[l,l,None], d[i,l,None])
        self.single_d = d[self.single,self.single]

    def pair_oracle(self, state, w):
        """Tangent the PSD remainder and minimize the disjunction in each block."""
        q = self.pair_q
        qw = q @ w
        g = self.c - qw
        t, box, penalty = self.t, self.box, self.lam
        i,l = self.pairs.T
        flags = np.array([[0,0],[0,1],[1,0],[1,1]])
        a = np.where(flags[:,0], box, t)
        b = np.where(flags[:,1], box, t)
        gx, gy = g[i,None], g[l,None]
        dx, dy, cross = self.pair_d

        def val(x, y):
            return x*(.5*dx*x-gx) + y*(.5*dy*y-gy) + cross*x*y

        result = val(a,b)
        bx = np.broadcast_to(a, result.shape).copy()
        by = np.broadcast_to(b, result.shape).copy()
        def consider(x, y, valid=True):
            v = val(x,y)
            better = valid & (v < result)
            np.copyto(result, v, where=better)
            np.copyto(bx, x, where=better)
            np.copyto(by, y, where=better)
        for sign in (-1,1):
            xx, yy = sign*a, sign*b
            consider(xx, b)
            consider(xx, -b)
            opty = np.clip((gy-cross*xx)/np.where(dy>0,dy,1.), -b,b)
            optx = np.clip((gx-cross*yy)/np.where(dx>0,dx,1.), -a,a)
            consider(xx, opty, dy>0)
            consider(optx, yy, dx>0)
        det = dx*dy-cross*cross
        den = np.where(det>0,det,1.)
        xx = (dy*gx-cross*gy)/den
        yy = (dx*gy-cross*gx)/den
        consider(xx, yy, (dx>0) & (det>0) & (abs(xx)<=a) & (abs(yy)<=b))
        result += penalty * flags.sum(axis=1)
        ok = ((state[i,None]<0)|(state[i,None]==flags[:,0])) & ((state[l,None]<0)|(state[l,None]==flags[:,1]))
        result = np.where(ok,result,np.inf)
        d = self.single_d
        g1 = np.abs(g[self.single])
        z0 = np.where(d>0,np.minimum(g1/np.where(d>0,d,1.),t),t)
        z1 = np.where(d>0,np.minimum(g1/np.where(d>0,d,1.),box),box)
        r0 = z0*(.5*d*z0-g1)
        r1 = z1*(.5*d*z1-g1) + penalty
        f = state[self.single]
        single = np.where(f==0,r0,np.where(f==1,r1,np.minimum(r0,r1)))
        sel = np.argmin(result,axis=1)
        rows = np.arange(i.size)
        on = (f == 1) | ((f == -1) & (r1 < r0))
        z = np.empty_like(w)
        active = np.empty(self.p,dtype=bool)
        z[i], z[l] = bx[rows,sel], by[rows,sel]
        active[i],active[l] = flags[sel].T
        z[self.single] = np.sign(g[self.single])*np.where(on,z1,z0)
        active[self.single] = on
        linear_min = np.sum(single) + np.sum(result[rows,sel])
        atom = linear_min - qw @ z
        lb = self.c0 - .5*(w @ qw) + linear_min - self.pad
        return float(lb), z, float(atom), active

    def optimize_pairs(self,state,start,steps):
        # Conditional gradients in the convex hull of the lifted block epigraphs.
        # Their dual bounds are valid at every center, not just at convergence.
        q = self.pair_q
        best, z, f, active = self.pair_oracle(state,start)
        self.fit(active)
        if best >= self.best-_GAP:
            return best
        atoms = [z]
        linear = [f]
        weights = [1.0]
        w = z.copy()
        for k in range(steps):
            lb,z,f,active = self.pair_oracle(state,w)
            best = max(best,lb)
            if k < 3 or k % 8 == 0:
                self.fit(active)
            if best >= self.best-_GAP:
                break
            grad = q @ w
            fw = f + grad @ z
            vals = np.asarray(atoms) @ grad + linear
            avg = np.dot(weights,vals)
            if avg-fw < 1e-8:
                break
            j = int(np.argmax(vals))
            # Prefer an away step when it gives more descent. A regular
            # Frank-Wolfe step introduces just one new atom.
            away = vals[j]-avg > avg-fw and weights[j] < 1.0
            if away:
                delta = w-atoms[j]
                slope = avg-vals[j]
                cap = weights[j]/(1-weights[j])
            else:
                delta = z-w
                slope = fw-avg
                cap = 1.0
            curv = delta @ q @ delta
            alpha = min(cap,-slope/curv) if curv > 0 else cap
            w += alpha*delta
            if away:
                weights = [(1+alpha)*v for v in weights]
                weights[j] -= alpha
                if alpha == cap:
                    del weights[j],atoms[j],linear[j]
            elif alpha == 1:
                weights,atoms,linear = [1.0],[z],[f]
            else:
                weights = [(1-alpha)*v for v in weights] + [alpha]
                atoms.append(z)
                linear.append(f)
        return best

    def objective(self, w):
        r = self.x @ w - self.y
        return float(0.5*(r @ r + self.ridge*(w @ w)) + self.lam*np.count_nonzero(np.abs(w) > _ZERO))

    def fit(self, active):
        key = tuple(active)
        if key in self.tried:
            return self.tried[key]
        limit = np.where(active, self.box, self.t)
        fit = lsq_linear(self.a, self.b, bounds=(-limit, limit), method='bvls', tol=1e-13)
        w = np.clip(fit.x, -limit, limit)
        self.tried[key] = w
        value = self.objective(w)
        if value < self.best:
            self.best, self.w = value, w
        return w

    def search_neighbors(self):
        # Local search only sets an upper bound; it never supplies a certificate.
        for _ in range(3):
            active = np.abs(self.w) > _ZERO
            prev = self.best
            for j in range(self.p):
                active[j] = not active[j]
                self.fit(active)
                active[j] = not active[j]
            if self.best >= prev:
                break

    def relaxation(self, state, w, steps):
        free = state == -1
        cap = np.where(state == 0, self.t, self.box)
        if self.box == self.t:
            free = np.zeros_like(free)
        fi = np.flatnonzero(free)
        q, c, d, step = self.q, self.c, self.d, self.step
        b, t, h = self.bridge, self.t, self.slope
        charge = self.lam*np.count_nonzero(state == 1)
        best, best_w, best_g = -np.inf, w, np.zeros_like(w)
        v = w = np.clip(w, -cap, cap)
        accel = 1.0
        for k in range(steps+1):
            if k % 8 == 0 or k == steps:
                qw = q @ w
                g = c - qw
                za = np.minimum(np.abs(g)/d, cap)
                parts = za*(0.5*d*za - np.abs(g))
                if fi.size:
                    z0 = np.minimum(za[fi], t)
                    parts[fi] = np.minimum(z0*(0.5*d*z0 - np.abs(g[fi])), parts[fi]+self.lam)
                lb = self.c0 - 0.5*(w @ qw) + charge + np.sum(parts) - self.pad
                if lb > best:
                    best, best_w, best_g = lb, w, g
                if lb >= self.best-_GAP:
                    break
                aw = np.abs(w)
                f = self.c0 + 0.5*(w @ qw) - c @ w + 0.5*d*(w @ w) + charge
                if fi.size:
                    z = aw[fi]
                    f += np.sum(np.where(z <= t, 0.0, np.where(z < b, 0.5*d*(t*t-z*z)+h*(z-t), self.lam)))
                if f-lb <= max(2*self.pad, 2e-8) or k == steps:
                    break
            z = v + step*(c-q @ v)
            w1 = np.clip(z/(1+step*d), -cap, cap)
            if fi.size:
                u = np.abs(z[fi]); plain = u/(1+step*d)
                a = np.where(u <= b+step*h, np.clip(u-step*h, t, b), np.maximum(plain, b))
                a = np.where(u <= t*(1+step*d), plain, a)
                w1[fi] = np.sign(z[fi])*np.minimum(a, self.box)
            if np.dot(v-w1, w1-w) > 0:
                accel = 1.0
                v = w1
            else:
                new_accel = 0.5*(1+np.sqrt(1+4*accel*accel))
                v = w1 + (accel-1)/new_accel*(w1-w)
                accel = new_accel
            w = w1
        return float(best), best_w, best_g


def solve_certified_box_l0(instance: dict) -> dict:
    """Return a feasible solution and the lower bound of a complete search cover."""
    prob = _Problem(instance)
    p = prob.p
    nodes = 0
    terminal = np.inf
    queue = []
    serial = count()

    def visit(state, start, parent_lb):
        nonlocal nodes, terminal
        nodes += 1
        free = state == -1
        if not np.any(free):
            start = prob.fit(state == 1)
        lb, w, g = prob.relaxation(state, start, 256 if np.any(free) else 2048)
        if prob.pairs.size and lb < prob.best-_GAP:
            lb = max(lb, prob.optimize_pairs(state, prob.w, 100 if nodes == 1 else 20))
        lb = max(0.0, lb, parent_lb)
        if lb < prob.best-_GAP and np.any(free):
            prob.fit((state == 1) | (free & (np.abs(w) > (prob.t+prob.bridge)/2)))
            if nodes == 1:
                prob.fit((state == 1) | (free & (np.abs(w) >= prob.bridge)))
                prob.fit(np.ones(p, dtype=bool))
                prob.search_neighbors()
        if lb >= prob.best-_GAP or not np.any(free):
            terminal = min(terminal, lb)
        else:
            heapq.heappush(queue, (lb, next(serial), state, w, g))

    visit(np.full(p, -1 if prob.box > prob.t else 0, dtype=np.int8), np.zeros(p), 0.0)
    while queue and queue[0][0] < prob.best-_GAP and nodes+2 <= 600:
        lb, _, state, w, g = heapq.heappop(queue)
        # In the envelope's linear segment the best indicators disagree with
        # the relaxed coefficient. Prefer a split exposing a large such gap.
        candidates = np.flatnonzero(state == -1)
        a = np.abs(w[candidates]); s = np.abs(g[candidates]); d = prob.d
        z0 = np.minimum(s/d, prob.t); z1 = np.minimum(s/d, prob.box)
        p0 = z0*(0.5*d*z0-s); p1 = z1*(0.5*d*z1-s)+prob.lam
        score = np.minimum(np.maximum(a-z0, 0)*d, np.maximum(z1-a, 0)*d)
        score += 1e-3*(np.abs(p1-p0)<1e-7)*a
        j = candidates[np.argmax(score)]
        first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
        for bit in (first, 1-first):
            child = state.copy()
            child[j] = bit
            visit(child, w, lb)
    lb = min(terminal, queue[0][0] if queue else np.inf, prob.best)
    if prob.best-lb > 1e-5:
        raise RuntimeError(f'Certificate gap {prob.best-lb:.6g} after {nodes} nodes')
    return {'weights': prob.w.tolist(), 'objective': float(prob.best),
            'certified_lower_bound': float(lb), 'nodes_explored': nodes}
