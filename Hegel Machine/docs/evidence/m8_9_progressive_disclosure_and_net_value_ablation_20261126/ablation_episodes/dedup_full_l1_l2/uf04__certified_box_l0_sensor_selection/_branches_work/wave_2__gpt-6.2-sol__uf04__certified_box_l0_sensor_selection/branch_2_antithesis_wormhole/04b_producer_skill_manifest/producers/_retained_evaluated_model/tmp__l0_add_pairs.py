from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = p.read_text()
s = s.replace('        self.tried = set()','        self.tried = set()\n        self.prepare_pairs(h)')
pos = s.index('    def objective(')
s = s[:pos] + '''    def prepare_pairs(self, h):
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
        q = off + np.diag(np.sum(np.abs(off), axis=1) + self.err_h)
        self.pair_q = rot @ q @ rot.T
        d = h - self.pair_q
        i,l = self.pairs.T
        self.pair_d = (d[i,i,None], d[l,l,None], d[i,l,None])
        self.single_d = d[self.single,self.single]

    def pair_bound(self, state, w):
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

        result = np.minimum(np.minimum(val(a,b),val(a,-b)), np.minimum(val(-a,b),val(-a,-b)))
        for sign in (-1,1):
            xx, yy = sign*a, sign*b
            opty = np.clip((gy-cross*xx)/np.where(dy>0,dy,1.), -b,b)
            optx = np.clip((gx-cross*yy)/np.where(dx>0,dx,1.), -a,a)
            result = np.minimum(result,np.where(dy>0,val(xx,opty),np.inf))
            result = np.minimum(result,np.where(dx>0,val(optx,yy),np.inf))
        det = dx*dy-cross*cross
        den = np.where(det>0,det,1.)
        xx = (dy*gx-cross*gy)/den
        yy = (dx*gy-cross*gx)/den
        interior = (dx>0) & (det>0) & (abs(xx)<=a) & (abs(yy)<=b)
        result = np.minimum(result,np.where(interior,val(xx,yy),np.inf))
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
        base = self.c0 - .5*(w @ qw) + np.sum(single) - self.pad
        return float(base + np.sum(result.min(axis=1)))

''' + s[pos:]
s = s.replace('        lb = max(0.0, lb, parent_lb)', '        if prob.pairs.size and lb < prob.best-_GAP:\n            lb = max(lb, prob.pair_bound(state,w))\n        lb = max(0.0, lb, parent_lb)')
p.write_text(s)
