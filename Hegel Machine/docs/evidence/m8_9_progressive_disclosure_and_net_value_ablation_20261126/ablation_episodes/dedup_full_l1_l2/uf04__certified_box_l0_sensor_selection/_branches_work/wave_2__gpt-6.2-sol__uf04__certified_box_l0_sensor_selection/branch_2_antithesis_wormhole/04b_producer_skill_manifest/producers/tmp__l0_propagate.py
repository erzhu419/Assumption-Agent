from pathlib import Path
p=Path('/holdout_task/certified_l0_solver.py')
s=p.read_text()
s=s.replace('def pair_oracle(self, state, w):','def pair_oracle(self, state, w, allowed):')
s=s.replace('        result = np.where(ok,result,np.inf)','        result = np.where(ok & allowed,result,np.inf)')
s=s.replace('        return float(lb), z, float(atom), active','        # Conditioning one block changes only its contribution to this tangent.\n        conditional = lb + (result - result[rows,sel,None])\n        return float(lb), z, float(atom), active, conditional')
s=s.replace('def optimize_pairs(self,state,start,steps):','def optimize_pairs(self,state,start,steps,allowed):')
s=s.replace('best, z, f, active = self.pair_oracle(state,start)','best, z, f, active, bounds = self.pair_oracle(state,start,allowed)')
s=s.replace('            return best\n','            return best, bounds\n',1)
s=s.replace('            lb,z,f,active = self.pair_oracle(state,w)','            lb,z,f,active,conditional = self.pair_oracle(state,w,allowed)\n            np.maximum(bounds,conditional,out=bounds)')
s=s.replace('        return best\n','        return best, bounds\n',1)
a=s.index('    def visit(')
b=s.index('    lb = min(terminal',a)
s=s[:a]+'''    flags = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.int8)
    pair_members = np.zeros(p,dtype=bool)
    pair_members[prob.pairs.ravel()] = True

    def visit(state, allowed, start, parent_lb):
        nonlocal nodes, terminal
        nodes += 1
        if allowed.size:
            state = state.copy()
            for k, indices in enumerate(prob.pairs):
                opts = flags[allowed[k]]
                fixed = np.all(opts == opts[0], axis=0)
                state[indices[fixed]] = opts[0,fixed]
        free = state == -1
        if not np.any(free):
            start = prob.fit(state == 1)
        lb, w, g = prob.relaxation(state, start, 256 if np.any(free) else 2048)
        lb = max(0.0, lb, parent_lb)
        bounds = None
        if allowed.size and lb < prob.best-_GAP:
            pair_lb,bounds = prob.optimize_pairs(state, prob.w, 100 if nodes == 1 else 30, allowed)
            lb = max(lb,pair_lb)
        if lb < prob.best-_GAP and np.any(free):
            prob.fit((state == 1) | (free & (np.abs(w) > (prob.t+prob.bridge)/2)))
            if nodes == 1:
                prob.fit((state == 1) | (free & (np.abs(w) >= prob.bridge)))
                prob.fit(np.ones(p, dtype=bool))
                prob.search_neighbors()
        if bounds is not None and lb < prob.best-_GAP:
            # Every deleted cell has its own certificate over all the other
            # blocks. Save the bound for that part of the original cover.
            np.maximum(bounds,lb,out=bounds)
            cut = allowed & (bounds >= prob.best-_GAP)
            if np.any(cut):
                terminal = min(terminal,float(np.min(bounds[cut])))
                allowed = allowed & ~cut
                if np.any(~allowed.any(axis=1)):
                    return
            # The remaining patterns also cover the current node in each block.
            lb = max(lb,float(np.max(np.min(np.where(allowed,bounds,np.inf),axis=1))))
        if lb >= prob.best-_GAP or not np.any(free):
            terminal = min(terminal, lb)
        else:
            heapq.heappush(queue, (lb, next(serial), state, allowed, bounds, w, g))

    allowed = np.ones((len(prob.pairs),4),dtype=bool)
    if prob.box == prob.t:
        allowed[:,1:] = False
    visit(np.full(p, -1 if prob.box > prob.t else 0, dtype=np.int8),allowed,np.zeros(p),0.0)
    while queue and queue[0][0] < prob.best-_GAP and nodes+2 <= 600:
        lb, _, state, allowed, bounds, w, g = heapq.heappop(queue)
        # Split off the best-certified group of pair patterns. When only the
        # two single activations survive, the split has exactly two children.
        choice = None
        if bounds is not None:
            for k, a in enumerate(allowed):
                for group in ((1,0,0,0),(0,0,0,1),(0,1,1,0),(0,1,0,0),(0,0,1,0)):
                    left = a & np.asarray(group,dtype=bool)
                    right = a & ~left
                    if not left.any() or not right.any():
                        continue
                    l,r = np.min(bounds[k,left]),np.min(bounds[k,right])
                    merit = (max(l,r),min(l,r))
                    if choice is None or merit > choice[0]:
                        choice = merit,k,left,right,l,r
        if choice is not None:
            _,k,left,right,l,r = choice
            for mask, lower in sorted(((left,l),(right,r)),key=lambda v:v[1]):
                child = allowed.copy()
                child[k] = mask
                visit(state,child,w,max(lb,lower))
            continue
        candidates = np.flatnonzero((state == -1) & ~pair_members)
        if not candidates.size:
            # Pair filtering can fix the last indicators without a split.
            visit(state,allowed,w,lb)
            continue
        a = np.abs(w[candidates]); sg = np.abs(g[candidates]); d = prob.d
        z0 = np.minimum(sg/d, prob.t); z1 = np.minimum(sg/d, prob.box)
        score = np.minimum(np.maximum(a-z0, 0)*d, np.maximum(z1-a, 0)*d)
        j = candidates[np.argmax(score)]
        first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
        for bit in (first, 1-first):
            child = state.copy()
            child[j] = bit
            visit(child,allowed,w,lb)
''' + s[b:]
p.write_text(s)
