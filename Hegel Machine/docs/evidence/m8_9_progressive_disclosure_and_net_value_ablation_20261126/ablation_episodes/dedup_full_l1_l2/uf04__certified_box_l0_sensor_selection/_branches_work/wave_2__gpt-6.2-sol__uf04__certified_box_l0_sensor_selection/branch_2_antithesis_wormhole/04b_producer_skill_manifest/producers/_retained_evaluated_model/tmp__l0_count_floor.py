from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = Path('/tmp/l0_v11.py').read_text()
s = s.replace('        self.prepare_pairs(h)', '        self.prepare_pairs(h)\n        self.count_floor = 0\n        self.count_probe = None')
a = s.index('    @staticmethod\n    def budget_minima')
b = s.index('    def pair_oracle', a)
s = s[:a] + '''    @staticmethod
    def budget_minima(cost, floor, limit, pair_count):
        """Block dynamic program with terminal bounds on total activations."""
        counts = np.array([0,1,1,2])
        used = np.arange(limit+1)[:,None]
        after = used + counts
        valid = after <= limit
        after = np.minimum(after, limit)
        m = len(cost)
        suffix = np.full((m+1,limit+1),np.inf)
        suffix[m,floor:] = 0.
        picks = np.empty((m,limit+1),dtype=int)
        for k in range(m-1,-1,-1):
            v = np.where(valid, cost[k] + suffix[k+1,after], np.inf)
            picks[k] = np.argmin(v,axis=1)
            suffix[k] = np.min(v,axis=1)
        selection = None
        if np.isfinite(suffix[0,0]):
            selection = np.empty(m,dtype=int)
            b = 0
            for k in range(m):
                j = picks[k,b]
                selection[k] = j
                b += counts[j]
        prefix = np.full(limit+1,np.inf)
        prefix[0] = 0.0
        conditional = np.empty((pair_count,4))
        for k in range(pair_count):
            v = np.where(valid, prefix[:,None] + cost[k] + suffix[k+1,after], np.inf)
            conditional[k] = v.min(axis=0)
            nxt = prefix + cost[k,0]
            if limit >= 1:
                nxt[1:] = np.minimum(nxt[1:],prefix[:-1] + min(cost[k,1],cost[k,2]))
            if limit >= 2:
                nxt[2:] = np.minimum(nxt[2:],prefix[:-2] + cost[k,3])
            prefix = nxt
        return suffix[0,0], selection, conditional

''' + s[b:]
a = s.index('        limit = min(self.p, int((self.best + self.pad)/penalty))')
b = s.index('        z = np.empty_like(w)', a)
s = s[:a] + '''        limit = self.count_probe
        if limit is None:
            limit = min(self.p, int((self.best + self.pad)/penalty))
        greedy_count = flags[sel].sum() + np.count_nonzero(on)
        if self.count_probe is not None or (lb < self.best-_GAP and (
                self.count_floor > 0 or (limit < self.p and greedy_count+2 > limit))):
            singles = np.full((len(f),4),np.inf)
            singles[:,0] = np.where(f != 1,r0,np.inf)
            singles[:,1] = np.where(f != 0,r1,np.inf)
            minimum, chosen, cells = self.budget_minima(
                np.vstack((result,singles)),self.count_floor,limit,len(rows))
            cutoff = np.inf if self.count_probe is not None else (limit+1)*penalty-self.pad
            lb = max(lb,min(base+minimum,cutoff))
            conditional = np.maximum(conditional,np.minimum(base+cells,cutoff))
            if chosen is not None:
                sel, on = chosen[:len(rows)], chosen[len(rows):] == 1
                linear_min = minimum
''' + s[b:]
a = s.index('        if nodes == 1 and bounds is not None')
s = s[:a] + '''        if nodes == 1 and bounds is not None and lb < prob.best-_GAP:
            # Prove a band of low activation counts cannot improve the incumbent.
            limit = min(p, int((prob.best + prob.pad)/prob.lam))
            if 1 <= limit < p:
                for cap in range(limit-1,max(-1,limit-3),-1):
                    prob.count_probe = cap
                    nodes += 1
                    proof,_ = prob.optimize_pairs(state,prob.w,240,allowed)
                    prob.count_probe = None
                    if proof >= prob.best-_GAP:
                        terminal = min(terminal,proof)
                        prob.count_floor = cap+1
                        break
''' + s[a:]
# Root is identified by its incoming LB and initialization, not the extra probe counter.
s = s.replace('        if nodes == 1 and bounds is not None and lb < prob.best-_GAP:\n            # Apply', '        if depth == 0 and bounds is not None and lb < prob.best-_GAP:\n            # Apply')
p.write_text(s)
