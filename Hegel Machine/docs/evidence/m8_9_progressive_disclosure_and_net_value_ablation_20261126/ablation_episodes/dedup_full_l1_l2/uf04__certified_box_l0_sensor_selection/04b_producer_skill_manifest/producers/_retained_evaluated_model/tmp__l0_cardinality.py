from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = Path('/tmp/l0_v10.py').read_text()
a = s.index('    def pair_oracle(')
s = s[:a] + '''    @staticmethod
    def budget_minima(cost, limit, pair_count):
        """Minimize a block tangent with an upper bound on total activations."""
        counts = np.array([0,1,1,2])
        remaining = np.arange(limit+1)[:,None] - counts
        valid = remaining >= 0
        remaining = np.maximum(remaining,0)
        m = len(cost)
        suffix = np.zeros((m+1,limit+1))
        picks = np.empty((m,limit+1),dtype=int)
        for k in range(m-1,-1,-1):
            v = np.where(valid, cost[k] + suffix[k+1,remaining], np.inf)
            picks[k] = np.argmin(v,axis=1)
            suffix[k] = np.min(v,axis=1)
        selection = None
        if np.isfinite(suffix[0,limit]):
            selection = np.empty(m,dtype=int)
            b = limit
            for k in range(m):
                j = picks[k,b]
                selection[k] = j
                b -= counts[j]
        prefix = np.full(limit+1,np.inf)
        prefix[0] = 0.0
        conditional = np.empty((pair_count,4))
        for k in range(pair_count):
            v = np.where(valid, prefix[:,None] + cost[k] + suffix[k+1,remaining[::-1]], np.inf)
            # remaining[::-1][i,j] is limit-i-counts[j].
            v = np.where(valid[::-1],v,np.inf)
            conditional[k] = v.min(axis=0)
            nxt = prefix + cost[k,0]
            if limit >= 1:
                nxt[1:] = np.minimum(nxt[1:],prefix[:-1] + min(cost[k,1],cost[k,2]))
            if limit >= 2:
                nxt[2:] = np.minimum(nxt[2:],prefix[:-2] + cost[k,3])
            prefix = nxt
        return suffix[0,limit], selection, conditional

''' + s[a:]
s = s.replace('v = np.where(valid, prefix[:,None] + cost[k] + suffix[k+1,remaining[::-1]], np.inf)\n            # remaining[::-1][i,j] is limit-i-counts[j].\n            v = np.where(valid[::-1],v,np.inf)', 'v = np.where(valid[::-1], prefix[:,None] + cost[k] + suffix[k+1,remaining[::-1]], np.inf)')
a = s.index('        sel = np.argmin(result,axis=1)')
b = s.index('    def optimize_pairs',a)
s = s[:a] + '''        sel = np.argmin(result,axis=1)
        rows = np.arange(i.size)
        on = (f == 1) | ((f == -1) & (r1 < r0))
        base = self.c0 - .5*(w @ qw) - self.pad
        linear_min = np.sum(single) + np.sum(result[rows,sel])
        lb = base + linear_min
        conditional = lb + (result - result[rows,sel,None])
        # Every omitted pattern costs at least (limit+1)*penalty. Keep
        # that part of the cover by capping both unconditional and
        # conditional bounds at the cost floor for omitted patterns.
        limit = min(self.p, int((self.best + self.pad)/penalty))
        greedy_count = flags[sel].sum() + np.count_nonzero(on)
        if limit < self.p and greedy_count+2 > limit and lb < self.best-_GAP:
            singles = np.full((len(f),4),np.inf)
            singles[:,0] = np.where(f != 1,r0,np.inf)
            singles[:,1] = np.where(f != 0,r1,np.inf)
            minimum, chosen, cells = self.budget_minima(np.vstack((result,singles)),limit,len(rows))
            cutoff = (limit+1)*penalty-self.pad
            lb = max(lb,min(base+minimum,cutoff))
            conditional = np.maximum(conditional,np.minimum(base+cells,cutoff))
            if chosen is not None:
                sel, on = chosen[:len(rows)], chosen[len(rows):] == 1
                linear_min = minimum
        z = np.empty_like(w)
        active = np.empty(self.p,dtype=bool)
        z[i], z[l] = bx[rows,sel], by[rows,sel]
        active[i],active[l] = flags[sel].T
        z[self.single] = np.sign(g[self.single])*np.where(on,z1,z0)
        active[self.single] = on
        atom = linear_min - qw @ z
        return float(lb), z, float(atom), active, conditional

''' + s[b:]
p.write_text(s)
