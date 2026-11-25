"""Certified branch-and-bound for box-constrained sparse ridge regression."""
import heapq
from itertools import combinations
import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.linalg import eigvalsh


_THRESHOLD = 1e-9
_GAP = 9.8e-6


class _Problem:
    def __init__(self, X, y, ridge, box, penalty):
        self.X, self.y = X, y
        self.box, self.pen, self.ridge = box, penalty, ridge
        self.t = min(box, _THRESHOLD)
        self.p = X.shape[1]
        self.H = X.T @ X + ridge * np.eye(self.p)
        self.c = X.T @ y
        self.ldX, self.ldy = X.astype(np.longdouble), y.astype(np.longdouble)
        # Use a diagonal minorant of the Gram matrix plus ridge. Certify the
        # spectral estimate from the reconstruction and orthogonality residuals.
        norms = np.sqrt(np.diag(self.H))
        corr = self.H / norms[:, None] / norms[None, :]
        s, Q = np.linalg.eigh(corr)
        orth = np.linalg.norm(Q.T @ Q - np.eye(self.p), np.inf)
        res = np.linalg.norm(corr - (Q * s) @ Q.T, np.inf)
        u = np.finfo(float).eps
        error = 32 * u * (X.shape[0] + self.p + 1) * max(1., np.linalg.norm(corr, np.inf))
        mu = s[0] * (1 - orth) - res - error
        self.d = .99 * mu * norms**2 if mu > 0 else np.full(self.p, .99 * ridge)
        self.K = self.H - np.diag(self.d)
        self.diag = np.diag(self.K).copy()
        self.beta = np.minimum(box, self.t + np.sqrt(2 * penalty / self.d))
        if box > self.t:
            self.slope = penalty / (self.beta-self.t) + .5*self.d*(self.beta+self.t)
        else:
            self.slope = np.full(self.p, np.inf)
        self.ldd = self.d.astype(np.longdouble)
        self.best = .5 * float(y @ y)
        self.w = np.zeros(self.p)

    def bound(self, x, state):
        # Tangent to the convex remainder: all support restrictions then
        # separate into two exact one-dimensional quadratic minima.
        x = x.astype(np.longdouble)
        r = self.ldX @ x - self.ldy
        g = self.ldX.T @ r + (self.ridge - self.ldd)*x
        constant = .5*(r @ r + self.ridge*(x @ x) - (self.ldd*x) @ x) - g @ x
        z = np.clip(-g/self.ldd, -self.t, self.t)
        a = np.clip(-g/self.ldd, -self.box, self.box)
        off = z*(g + .5*self.ldd*z)
        on = self.pen + a*(g + .5*self.ldd*a)
        terms = np.where(state == 0, off, np.where(state == 1, on, np.minimum(off,on)))
        # Budget roundoff against the magnitudes before cancellation.
        absX = np.abs(self.ldX)
        v = absX @ np.abs(x) + np.abs(self.ldy)
        mag = (v @ v + self.box*(v @ absX).sum() + (self.ridge+self.ldd.max())*self.p*self.box**2 + self.p*self.pen)
        pad = 16*np.finfo(np.longdouble).eps*(len(r)+self.p+1)*mag
        lower = np.nextafter(float(constant + terms.sum() - pad), -np.inf)
        return lower, np.asarray(off-terms, float), np.asarray(on-terms, float), on < off

    def fit_fixed(self, mask):
        lim = np.where(mask, self.box, self.t)
        augmented = np.vstack((self.X, np.sqrt(self.ridge)*np.eye(self.p)))
        target = np.r_[self.y, np.zeros(self.p)]
        return np.clip(lsq_linear(augmented, target, bounds=(-lim, lim),
                                  method='bvls', tol=1e-13).x, -lim, lim)

    def improve(self, x, mask=None):
        if mask is not None:
            x = self.fit_fixed(mask)
        r = self.X @ x - self.y
        value = .5*(r @ r + self.ridge*(x @ x)) + self.pen*np.count_nonzero(np.abs(x) > _THRESHOLD)
        if value < self.best:
            self.best, self.w = float(value), x.copy()

    def relax(self, state, start):
        # The convex envelope is the ridge quadratic on [0,t], its common
        # supporting chord up to beta, and the paid quadratic thereafter.
        x = start.copy()
        lim = np.where(state == 0, self.t, self.box)
        np.clip(x, -lim, lim, out=x)
        g = self.K @ x - self.c
        free = state == -1
        for sweep in range(220):
            move = 0.
            for j in range(self.p):
                h = g[j] - self.diag[j]*x[j]
                m = abs(h)
                v = m / self.H[j,j]
                if free[j] and v > self.t:
                    l = self.slope[j]
                    if m <= l + self.diag[j]*self.t:
                        v = self.t
                    elif m < l + self.diag[j]*self.beta[j]:
                        v = (m-l)/self.diag[j]
                    else:
                        v = max(self.beta[j], v)
                new = -np.copysign(min(v, lim[j]), h)
                step = new-x[j]
                if step:
                    x[j] = new
                    g += step*self.K[:,j]
                    move = max(move, abs(step))
            if move < 2e-10*(1+np.max(np.abs(x))):
                break
        return x

    def search(self, tolerance=_GAP, budget=600):
        root = np.full(self.p, -1 if self.box > self.t else 0, dtype=np.int8)
        pq = [(-np.inf, 0, root, self.w.copy())]
        serial, nodes = 0, 0
        dropped = np.inf
        while pq and nodes < budget:
            inherited, _, state, x = heapq.heappop(pq)
            if inherited >= self.best - tolerance:
                dropped = min(dropped, inherited)
                break
            nodes += 1
            while True:
                x = self.relax(state, x)
                lb, off, on, preferred = self.bound(x, state)
                lb = max(inherited, lb)
                free = state == -1
                if lb < self.best - tolerance:
                    self.improve(x, (state == 1) | (free & (np.abs(x) > (self.t+self.beta)/2)))
                    alt = (state == 1) | (free & preferred)
                    if lb < self.best - tolerance and (nodes == 1 or not free.any() or isinstance(self, _PairedProblem)):
                        self.improve(x, alt)
                if lb >= self.best - tolerance:
                    dropped = min(dropped, lb)
                    break
                if not free.any():
                    x = self.fit_fixed(state == 1)
                    self.improve(x)
                    lb = max(lb, self.bound(x, state)[0])
                    if lb < self.best - tolerance:
                        raise RuntimeError('box subproblem did not close its dual gap')
                    dropped = min(dropped, lb)
                    break
                # Child increments originate at the raw tangent, not at the
                # inherited bound; the latter remains valid on either child.
                raw, _, _, _ = self.bound(x, state) if lb == inherited else (lb, None, None, None)
                lo = np.maximum(inherited, np.nextafter(raw + off, -np.inf))
                hi = np.maximum(inherited, np.nextafter(raw + on, -np.inf))
                # Fix indicators whose other choice already misses the cutoff.
                cut = self.best - tolerance
                kill0, kill1 = (lo >= cut) & free, (hi >= cut) & free
                if (kill0 | kill1).any():
                    dropped = min(dropped, np.min(lo[kill0], initial=np.inf), np.min(hi[kill1], initial=np.inf))
                    if (kill0 & kill1).any():
                        break
                    state[kill0], state[kill1] = 1, 0
                    inherited = max(lb, np.max(hi[kill0], initial=lb), np.max(lo[kill1], initial=lb))
                    continue
                # A large envelope defect favors branching on a genuinely
                # fractional coordinate, rather than a nearly decided one.
                ax = np.abs(x)
                chord = free & (ax > self.t) & (ax < self.beta)
                defect = np.zeros(self.p)
                defect[chord] = self.pen + .5*self.d[chord]*(ax[chord]**2-self.t**2) - self.slope[chord]*(ax[chord]-self.t)
                j = int(np.argmax(np.where(free, defect, -np.inf)))
                for val, b in ((0, lo[j]), (1, hi[j])):
                    child = state.copy()
                    child[j] = val
                    serial += 1
                    heapq.heappush(pq, (max(lb, b), serial, child, x))
                break
        lower = min(dropped, *( [pq[0][0]] if pq else [np.inf] ))
        if self.best - lower > tolerance + 2e-7 or not np.isfinite(lower):
            raise RuntimeError(f'certificate did not close: {nodes} nodes, gap {self.best-lower:.6g}')
        return self.w, self.best, lower, nodes



class _PairedProblem(_Problem):
    """Keep the joint choice and curvature of strongly correlated pairs."""

    def setup_pairs(self):
        norms = np.sqrt(np.diag(self.H))
        scores = np.abs(self.H / norms[:, None] / norms[None, :])
        np.fill_diagonal(scores, 0.)
        pairs = []
        unused = set(range(self.p))
        while scores.max() >= .85:
            j, k = np.unravel_index(scores.argmax(), scores.shape)
            pairs.append((j, k))
            unused.difference_update((j, k))
            scores[[j,k], :] = 0.
            scores[:, [j,k]] = 0.
        if not pairs or self.box <= self.t:
            return False
        B = np.zeros_like(self.H)
        for group in pairs + [(j,) for j in sorted(unused)]:
            ix = np.ix_(group, group)
            B[ix] = self.H[ix]
        fraction = .995 * eigvalsh(self.H, B, subset_by_index=[0,0])[0]
        if fraction <= 0 or np.min(fraction*norms**2/self.d) < 2.:
            return False
        B *= fraction
        # Certify the PSD remainder by Cholesky reconstruction plus a
        # positive ridge reserve; the generalized eigenvalue is only a proposal.
        reserve = .001 * self.ridge
        K = self.H - B
        try:
            L = np.linalg.cholesky(K - reserve*np.eye(self.p)).astype(np.longdouble)
        except np.linalg.LinAlgError:
            return False
        lB = B.astype(np.longdouble)
        ldK = self.ldX.T @ self.ldX + self.ridge*np.eye(self.p, dtype=np.longdouble) - lB
        error = np.max(np.sum(np.abs(ldK - reserve*np.eye(self.p) - L@L.T), axis=1))
        absX = np.abs(self.ldX)
        size = np.max(np.sum(absX.T@absX + abs(lB) + abs(L)@abs(L).T, axis=1)) + self.ridge
        error += 16*np.finfo(np.longdouble).eps*(len(self.y)+self.p+1)*size
        if error >= reserve:
            return False
        self.pairs = np.array(pairs)
        self.single = np.array(sorted(unused), dtype=int)
        self.B, self.ldB, self.K = B, lB, K
        self.codes = np.array([[0,0],[0,1],[1,0],[1,1]])
        return True

    def _modes(self, g, state, certified=False):
        # Each pair's convex box minimum occurs at its unconstrained
        # stationary point or on a face. Include all four faces.
        j,k = self.pairs.T
        B = self.ldB if certified else self.B
        a,b,c = (B[j,j,None], B[j,k,None], B[k,k,None])
        u,v = g[j,None], g[k,None]
        lx,ly = np.where(self.codes.T[:,None,:], self.box, self.t)
        det = a*c-b*b
        x0 = np.clip((b*v-c*u)/det,-lx,lx)
        y0 = np.clip((b*u-a*v)/det,-ly,ly)
        xx = np.broadcast_arrays(x0,-lx,lx,np.clip((-u+b*ly)/a,-lx,lx),np.clip((-u-b*ly)/a,-lx,lx))
        yy = np.broadcast_arrays(y0,np.clip((-v+b*lx)/c,-ly,ly),np.clip((-v-b*lx)/c,-ly,ly),-ly,ly)
        xx,yy = np.stack(xx),np.stack(yy)
        vals = xx*(u+.5*a*xx+b*yy)+yy*(v+.5*c*yy)
        ix = vals.argmin(axis=0)[None,:,:]
        xx,yy,vals = [np.take_along_axis(z, ix, axis=0)[0] for z in (xx,yy,vals)]
        if certified:
            dx,dy = u+a*xx+b*yy,v+b*xx+c*yy
            vals = vals - dx*xx - abs(dx)*lx - dy*yy - abs(dy)*ly
        vals += self.pen*self.codes.sum(axis=1)
        keep = np.all((state[self.pairs,None] == -1) | (state[self.pairs,None] == self.codes.T[None,:,:]), axis=1)
        vals = np.where(keep, vals, np.inf)
        h = self.single
        lim = np.array([self.t,self.box]); s = g[h,None]
        z = np.clip(-s/B[h,h,None],-lim,lim)
        sc = z*(s+.5*B[h,h,None]*z)+np.array([0.,self.pen])
        if certified:
            grad = s+B[h,h,None]*z
            sc -= grad*z+abs(grad)*lim
        sc = np.where((state[h,None] == -1) | (state[h,None] == np.arange(2)),sc,np.inf)
        return vals,sc,xx,yy,z

    def relax(self, state, start):
        if not np.any(state == -1):
            return self.fit_fixed(state == 1)
        j,k = self.pairs.T
        target = self.c
        def fg(x):
            g = self.K@x-target
            pair,single,xx,yy,z = self._modes(g,state)
            losses = []
            probs = []
            for a in (pair,single):
                mn = a.min(axis=1,keepdims=True)
                e = np.exp((mn-a)/temp)
                norm = e.sum(axis=1,keepdims=True)
                probs.append(e/norm)
                losses.append(mn.sum()-temp*np.log(norm).sum())
            zbar = np.empty_like(x)
            zbar[j] = (probs[0]*xx).sum(axis=1)
            zbar[k] = (probs[0]*yy).sum(axis=1)
            zbar[self.single] = (probs[1]*z).sum(axis=1)
            return .5*x@(self.K@x)-sum(losses), self.K@(x-zbar)
        x = start
        for temp in (max(1e-7, .006*self.pen), max(1e-8, .00006*self.pen), 1e-8):
            x = minimize(fg,x,jac=True,method='L-BFGS-B',bounds=[(-self.box,self.box)]*self.p,
                         options={'ftol':2e-14,'gtol':1e-8,'maxiter':140}).x
        return x

    def bound(self, x, state):
        x = x.astype(np.longdouble)
        r = self.ldX@x-self.ldy
        g = self.ldX.T@r+self.ridge*x-self.ldB@x
        constant = .5*(r@r+self.ridge*(x@x)-x@self.ldB@x)-g@x
        pair,single,xx,yy,z = self._modes(g,state,certified=True)
        bp,bs = pair.min(axis=1),single.min(axis=1)
        delta = np.zeros((2,self.p), dtype=np.longdouble)
        preferred = np.zeros(self.p,dtype=bool)
        preferred[self.pairs] = self.codes[pair.argmin(axis=1)]
        preferred[self.single] = single.argmin(axis=1)
        delta[:,self.single] = (single-bs[:,None]).T
        for q in range(2):
            for s in range(2):
                delta[s,self.pairs[:,q]] = pair[:,self.codes[:,q]==s].min(axis=1)-bp
        v = abs(self.ldX)@abs(x)+abs(self.ldy)
        scale = v@v + self.box*np.sum(v@abs(self.ldX)) + (self.ridge*self.p+abs(self.ldB).sum())*self.box**2+self.p*self.pen
        pad = 64*np.finfo(np.longdouble).eps*(len(r)+self.p+1)*scale
        lower = np.nextafter(float(constant+bp.sum()+bs.sum()-pad),-np.inf)
        return lower,delta[0],delta[1],preferred

class _TripleProblem(_PairedProblem):
    """Joint disjunctions for complete groups of three interchangeable columns."""

    def setup_triples(self):
        from itertools import product
        if self.p < 3 or self.box <= self.t:
            return False
        norms = np.sqrt(self.H.diagonal())
        corr = np.abs(self.H / norms[:, None] / norms[None, :])
        unused = set(range(self.p))
        groups = []
        while len(unused) >= 3:
            j, k, l = max(combinations(sorted(unused), 3),
                          key=lambda t: min(corr[t[0],t[1]], corr[t[0],t[2]], corr[t[1],t[2]]))
            if min(corr[j,k], corr[j,l], corr[k,l]) < .85:
                break
            groups.append((j,k,l))
            unused.difference_update((j,k,l))
        if not groups:
            return False
        while len(unused) >= 2:
            j,k = max(combinations(sorted(unused),2), key=lambda q: corr[q])
            if corr[j,k] < .85:
                break
            groups.append((j,k))
            unused.difference_update((j,k))
        B = np.zeros_like(self.H)
        for g in groups + [(i,) for i in sorted(unused)]:
            ix = np.ix_(g,g)
            B[ix] = self.H[ix]
        fraction = .995 * eigvalsh(self.H, B, subset_by_index=[0,0])[0]
        if fraction <= 0 or np.min(fraction * norms**2 / self.d) < 2.:
            return False
        B *= fraction
        K = self.H - B
        reserve = .001 * self.ridge
        try:
            L = np.linalg.cholesky(K-reserve*np.eye(self.p)).astype(np.longdouble)
        except np.linalg.LinAlgError:
            return False
        lB = B.astype(np.longdouble)
        absX = abs(self.ldX)
        D = self.ldX.T@self.ldX + (self.ridge-reserve)*np.eye(self.p) - lB - L@L.T
        scale = np.max(np.sum(absX.T@absX + abs(lB) + abs(L)@abs(L).T,axis=1)) + self.ridge
        err = np.max(np.sum(abs(D),axis=1)) + 16*np.finfo(np.longdouble).eps*(len(self.y)+self.p+1)*scale
        if err >= reserve:
            return False
        self.groups = np.array([g + (self.p,)*(3-len(g)) for g in groups])
        self.real = self.groups != self.p
        self.single = np.array(sorted(unused),dtype=int)
        self.B, self.ldB, self.K = B, lB, K
        padded = np.zeros((self.p+1,self.p+1))
        padded[:self.p,:self.p] = B
        padded[self.p,self.p] = 1.
        self.block = padded[self.groups[:,:,None],self.groups[:,None,:]]
        self.codes = np.array(list(product((0,1), repeat=3)))
        self.limits = np.where(self.real[:,None,:], np.where(self.codes,self.box,self.t),0.)
        faces = np.array(list(product((0,-1,1),repeat=3)))
        self.offset = np.zeros((len(groups),len(faces),len(self.codes),3))
        self.linear = np.zeros((len(groups),len(faces),3,3))
        for k, face in enumerate(faces):
            f, a = np.flatnonzero(face==0), np.flatnonzero(face!=0)
            for b, H in enumerate(self.block):
                v = self.limits[b]*face
                self.offset[b,k] = v
                if len(f):
                    inv = np.linalg.inv(H[np.ix_(f,f)])
                    self.linear[b,k][np.ix_(f,f)] = -inv
                    self.offset[b,k][:,f] = -v[:,a] @ H[np.ix_(a,f)] @ inv.T
        return True

    def _triple_modes(self, g, state, certified=False):
        h = np.r_[g,0][self.groups]
        # Face solves provide candidate points. In certification, a box tangent
        # bounds each mode regardless of the accuracy of its candidate point.
        v = self.offset + np.einsum('bfij,bj->bfi', self.linear, h.astype(float))[:,:,None,:]
        v = np.clip(v, -self.limits[:, None, :, :], self.limits[:, None, :, :])
        grad = np.einsum('bij,bfcj->bfci',self.block,v)
        a = np.sum(v*(h.astype(float)[:,None,None,:]+.5*grad),axis=-1)
        ix = a.argmin(axis=1)
        z = v[np.arange(len(h))[:,None],ix,np.arange(len(self.codes))[None,:]]
        if certified:
            z = z.astype(np.longdouble)
            B = self.block.astype(np.longdouble)
        else:
            B = self.block
        q = np.einsum('bij,bcj->bci',B,z)
        a = np.sum(z*(h[:,None,:]+.5*q),axis=-1)
        if certified:
            a -= np.sum((q+h[:,None,:])*z + abs(q+h[:,None,:])*self.limits,axis=-1)
        a += self.pen*self.codes.sum(axis=1)
        st = np.r_[state,0][self.groups][:,None,:]
        a = np.where(np.all((st == -1)|(st == self.codes),axis=-1),a,np.inf)
        j = self.single
        lim = np.array([self.t,self.box])
        d = (self.ldB if certified else self.B)[j,j,None]
        hj = g[j,None]
        u = np.clip(-hj/d,-lim,lim)
        b = u*(hj+.5*d*u)+np.array([0.,self.pen])
        if certified:
            b -= (hj+d*u)*u + abs(hj+d*u)*lim
        b = np.where((state[j,None]==-1)|(state[j,None]==np.arange(2)),b,np.inf)
        return a,b,z,u

    def relax(self, state, start):
        if not np.any(state == -1):
            return self.fit_fixed(state == 1)
        def fg(x):
            a,b,z,u = self._triple_modes(self.K@x-self.c,state)
            value = .5*x@(self.K@x)
            w = np.empty_like(x)
            for costs, pts, ix in ((a,z,self.groups),(b,u,self.single)):
                d = costs.min(axis=1,keepdims=True)
                pr = np.exp((d-costs)/temp)
                tot = pr.sum(axis=1,keepdims=True)
                value -= d.sum()-temp*np.log(tot).sum()
                avg = np.einsum('bc,bc...->b...',pr/tot,pts)
                if ix.ndim == 2:
                    w[ix[self.real]] = avg[self.real]
                else:
                    w[ix] = avg
            return value,self.K@(x-w)
        x = start
        for temp in (max(1e-7,.006*self.pen),max(1e-8,.00006*self.pen),1e-8):
            x = minimize(fg,x,jac=True,method='L-BFGS-B',bounds=[(-self.box,self.box)]*self.p,
                         options={'ftol':2e-14,'gtol':1e-8,'maxiter':120}).x
        return x

    def bound(self, x, state):
        x = x.astype(np.longdouble)
        r = self.ldX@x-self.ldy
        g = self.ldX.T@r+self.ridge*x-self.ldB@x
        constant = .5*(r@r+self.ridge*(x@x)-x@self.ldB@x)-g@x
        a,b,_,_ = self._triple_modes(g,state,certified=True)
        aa,bb = a.min(axis=1),b.min(axis=1)
        delta = np.zeros((2,self.p),dtype=np.longdouble)
        preferred = np.zeros(self.p,dtype=bool)
        preferred[self.groups[self.real]] = self.codes[a.argmin(axis=1)][self.real]
        preferred[self.single] = b.argmin(axis=1)
        delta[:,self.single] = (b-bb[:,None]).T
        for q in range(3):
            for s in range(2):
                valid = self.real[:,q]
                delta[s,self.groups[valid,q]] = a[valid][:,self.codes[:,q]==s].min(axis=1)-aa[valid]
        v = abs(self.ldX)@abs(x)+abs(self.ldy)
        scale = v@v + self.box*np.sum(v@abs(self.ldX)) + (self.ridge*self.p+abs(self.ldB).sum())*self.box**2+self.p*self.pen
        pad = 64*np.finfo(np.longdouble).eps*(len(r)+self.p+1)*scale
        lower = np.nextafter(float(constant+aa.sum()+bb.sum()-pad),-np.inf)
        return lower,delta[0],delta[1],preferred

class _CardinalityProblem(_Problem):
    """Branch on activation count, then on indicators within surviving counts."""

    def search(self, tolerance=_GAP, budget=600):
        from scipy.special import expit

        p = self.p
        root = np.full(p, -1, dtype=np.int8)
        pq = [(-np.inf, k, k, root.copy(), np.zeros(p+1)) for k in range(p+1)]
        serial, nodes, closed = p, 0, np.inf
        order = np.argsort(-abs(self.c))
        for k in range(p+1):
            mask = np.zeros(p, dtype=bool)
            mask[order[:k]] = True
            self.improve(self.w, mask)

        while pq and nodes < budget:
            if pq[0][0] >= self.best-tolerance:
                break
            inherited, _, count, state, point = heapq.heappop(pq)
            nodes += 1
            free = np.flatnonzero(state == -1)
            left = count-np.count_nonzero(state == 1)
            if left == 0 or left == len(free):
                state[free] = int(left > 0)
                x = self.fit_fixed(state == 1)
                self.improve(x)
                lb = max(inherited, self.bound(x,state)[0])
                closed = min(closed, lb)
                continue

            def fg(point):
                x, mu = point[:-1], point[-1]
                g = self.K@x-self.c
                v = np.clip(-g/self.d, -self.t, self.t)
                w = np.clip(-g/self.d, -self.box, self.box)
                off = v*(g+.5*self.d*v)
                on = self.pen+w*(g+.5*self.d*w)
                diff = (off[free]-on[free]-mu)/temp
                prob = expit(diff)
                z = np.where(state == 1,w,v)
                z[free] += prob*(w[free]-v[free])
                cost = np.where(state == 1,on,off).sum()-temp*np.logaddexp(0,diff).sum()
                return .5*x@(self.K@x)-cost+left*mu, np.r_[self.K@(x-z),left-prob.sum()]

            def evaluate(point):
                base, off, on, _ = self.bound(point[:-1],state)
                off, on = off[free].astype(np.longdouble), on[free].astype(np.longdouble)
                idx = np.argsort(on-off)
                rank = np.empty_like(idx)
                rank[idx] = np.arange(len(idx))
                ext = np.where(rank < left, on, off)
                # Bound the rounding of the individual activation increments as
                # well as that of their sum, including candidate child bounds.
                pad = 8*np.finfo(float).eps*(abs(base)+np.maximum(abs(on),abs(off)).sum())
                raw = np.longdouble(base)+ext.sum()-pad
                return raw, off, on, idx, rank

            raw, off, on, idx, rank = evaluate(point)
            if raw < self.best-tolerance:
                for fraction in (2e-3, 2e-5, 1e-7):
                    temp = max(1e-12,fraction*self.pen)
                    trial = minimize(fg,point,jac=True,method='L-BFGS-B',
                                     bounds=[(-self.box,self.box)]*p+[(None,None)],
                                     options={'maxiter':220,'ftol':1e-15,'gtol':1e-9}).x
                    updated = evaluate(trial)
                    if updated[0] > raw:
                        raw,off,on,idx,rank = updated
                        best_point = trial
                    elif fraction == 2e-3:
                        best_point = point
                    point = trial
                    if raw >= self.best-tolerance:
                        break
                point = best_point
            lb = max(inherited, float(np.nextafter(float(raw),-np.inf)))
            mask = state == 1
            mask[free[idx[:left]]] = True
            if lb < self.best-tolerance:
                self.improve(point[:-1],mask)
            if lb >= self.best-tolerance:
                closed = min(closed,lb)
                continue

            diff = on-off
            lo = np.where(rank < left, diff[idx[left]]-diff, 0)
            hi = np.where(rank >= left, diff-diff[idx[left-1]], 0)
            # Prioritize the activation most ambiguous at the optimized dual.
            j = int(np.argmin(abs(diff+point[-1])))
            for val, inc in ((0,lo[j]),(1,hi[j])):
                child = state.copy()
                child[free[j]] = val
                serial += 1
                b = max(lb,float(np.nextafter(float(raw+inc),-np.inf)))
                heapq.heappush(pq,(b,serial,count,child,point))
        lower = min(closed,pq[0][0] if pq else np.inf)
        if not np.isfinite(lower) or self.best-lower > tolerance+2e-7:
            raise RuntimeError(f'certificate did not close: {nodes} nodes, gap {self.best-lower:.6g}')
        return self.w,self.best,lower,nodes

def _components(X):
    """Partition columns by exact overlap of nonzero rows."""
    incidence = X != 0
    remaining = np.ones(X.shape[1], dtype=bool)
    while remaining.any():
        cols = np.zeros_like(remaining)
        cols[np.flatnonzero(remaining)[0]] = True
        while True:
            rows = incidence[:, cols].any(axis=1)
            more = incidence[rows].any(axis=0) & remaining & ~cols
            if not more.any():
                break
            cols |= more
        remaining &= ~cols
        yield rows, cols


def _identical_columns(X, y, ridge, box, penalty):
    """Reduce signed identical columns to two means for each paid count."""
    n, p = X.shape
    if p < 2:
        return None
    col = X[:, 0]
    same = np.all(X == col[:, None], axis=0)
    flipped = np.all(X == -col[:, None], axis=0)
    if not np.all(same | flipped):
        return None
    sign = np.where(same, 1., -1.)
    t = min(box, _THRESHOLD)
    col_ld, y_ld = col.astype(np.longdouble), y.astype(np.longdouble)
    lower, best, w = np.inf, np.inf, None
    # For a fixed paid count, symmetry and Jensen's inequality reduce its
    # convex box subproblem to one shared value per indicator class.
    for paid in range(p + 1 if box > t else 1):
        counts = np.array([p-paid, paid]); present = counts > 0
        m = counts[present]; lim = np.array([t, box])[present]
        D = col[:, None]*m
        A = np.vstack((D, np.diag(np.sqrt(ridge*m))))
        target = np.r_[y, np.zeros(len(m))]
        v = np.clip(lsq_linear(A, target, bounds=(-lim, lim), method='bvls',
                              tol=1e-13).x, -lim, lim)
        trial = sign*np.repeat(v, m)
        residual = X@trial-y
        value = .5*(residual@residual + ridge*(trial@trial)) + penalty*np.count_nonzero(abs(trial)>_THRESHOLD)
        if value < best:
            best, w = float(value), trial
        z = v.astype(np.longdouble)
        r = col_ld*(m@z)-y_ld
        grad = m*(col_ld@r+ridge*z)
        base = .5*(r@r + ridge*(m@(z*z))) + paid*np.longdouble(penalty)
        # Tangent minimization on the box; no accuracy assumption on BVLS.
        q = min(box, max(lim))
        env = abs(col_ld)*(m@lim.astype(np.longdouble)) + abs(y_ld)
        scale = env@env + p*ridge*np.longdouble(q)**2 + p*penalty
        pad = 32*np.finfo(np.longdouble).eps*(n+p+1)*scale
        lower = min(lower, np.nextafter(float(base-grad@z-abs(grad)@lim-pad), -np.inf))
    return w, best, lower, 1


def solve_certified_box_l0(instance: dict) -> dict:
    """Return feasible weights and a certified global objective interval."""
    X = np.asarray(instance['X'], dtype=float)
    y = np.asarray(instance['y'], dtype=float)
    ridge, box, penalty = (float(instance[k]) for k in ('ridge', 'box', 'l0_penalty'))
    w = np.zeros(X.shape[1])
    used = np.zeros(len(y), dtype=bool)
    lower, nodes = np.longdouble(0), 0
    blocks = list(_components(X))
    for rows, cols in blocks:
        if not rows.any():
            continue  # An identically zero feature is optimal at zero.
        xx, yy = X[np.ix_(rows, cols)], y[rows]
        result = _identical_columns(xx, yy, ridge, box, penalty)
        if result is None:
            problem = _TripleProblem(xx, yy, ridge, box, penalty)
            norms = np.sqrt(problem.H.diagonal())
            corr = abs(problem.H / norms[:, None] / norms[None, :])
            if problem.p >= 3 and box > problem.t and np.min(corr) >= .85:
                problem = _CardinalityProblem(xx, yy, ridge, box, penalty)
            elif not problem.setup_triples():
                problem = _PairedProblem(xx, yy, ridge, box, penalty)
                if not problem.setup_pairs():
                    problem = _Problem(xx, yy, ridge, box, penalty)
            result = problem.search(tolerance=_GAP/len(blocks), budget=600-nodes)
        w[cols], _, lo, count = result
        lower += np.longdouble(lo)
        nodes += count
        used |= rows
    unused_y = y[~used].astype(np.longdouble)
    tail = .5*(unused_y@unused_y)
    lower += tail*(1-4*np.finfo(np.longdouble).eps*(len(unused_y)+1))
    lower = float(np.nextafter(float(lower), -np.inf))
    residual = X@w-y
    value = .5*(residual@residual + ridge*(w@w)) + penalty*np.count_nonzero(abs(w)>_THRESHOLD)
    if value-lower > 1e-5 or nodes > 600 or not np.isfinite(lower):
        raise RuntimeError(f'certificate did not close: {nodes} nodes, gap {value-lower:.6g}')
    return {'weights': w.tolist(), 'objective': float(value), 'certified_lower_bound': lower, 'nodes_explored': max(1,nodes)}
