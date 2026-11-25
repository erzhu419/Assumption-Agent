from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = p.read_text().replace('def pair_bound(self, state, w):', 'def pair_oracle(self, state, w):')
a = s.index('        result = np.minimum(np.minimum(val(a,b)')
b = s.index('        result += penalty', a)
s = s[:a] + '''        result = val(a,b)
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
''' + s[b:]
a = s.index('        base = self.c0 - .5*(w @ qw)')
b = s.index('    def objective', a)
s = s[:a] + '''        sel = np.argmin(result,axis=1)
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

''' + s[b:]
s=s.replace('lb = max(lb, prob.pair_bound(state,w), prob.pair_bound(state,prob.w))', 'lb = max(lb, prob.optimize_pairs(state, prob.w, 100 if nodes == 1 else 20))')
a=s.index('        pairs = prob.pairs[np.all(state[prob.pairs]')
b=s.index('    lb = min(terminal',a)
s=s[:a]+'''        first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
        for bit in (first, 1-first):
            child = state.copy()
            child[j] = bit
            visit(child, w, lb)
'''+s[b:]
p.write_text(s)
