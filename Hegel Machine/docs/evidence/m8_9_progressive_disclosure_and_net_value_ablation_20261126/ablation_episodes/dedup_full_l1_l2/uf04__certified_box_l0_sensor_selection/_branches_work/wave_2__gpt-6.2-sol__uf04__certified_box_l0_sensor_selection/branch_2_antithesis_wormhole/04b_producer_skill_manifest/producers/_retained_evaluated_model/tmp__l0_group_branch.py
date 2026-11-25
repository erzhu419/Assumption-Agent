from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = p.read_text().replace('def pair_bound(self, state, w):', 'def pair_bound(self, state, w, exclusive):')
s = s.replace('        result = np.where(ok,result,np.inf)', '        ok &= ~exclusive[:, None] | (flags.sum(axis=1) == 1)\n        result = np.where(ok,result,np.inf)')
s = s.replace('def visit(state, start, parent_lb):', 'def visit(state, exclusive, start, parent_lb):')
s = s.replace('prob.pair_bound(state,w), prob.pair_bound(state,prob.w)', 'prob.pair_bound(state,w,exclusive), prob.pair_bound(state,prob.w,exclusive)')
s = s.replace('state, w, g))','state, exclusive, w, g))')
s = s.replace('dtype=np.int8), np.zeros(p), 0.0)', 'dtype=np.int8), np.zeros(len(prob.pairs), dtype=bool), np.zeros(p), 0.0)')
s = s.replace('lb, _, state, w, g = heapq.heappop(queue)', 'lb, _, state, exclusive, w, g = heapq.heappop(queue)')
a = s.index('        j = candidates[np.argmax(score)]')
b = s.index('    lb = min(terminal',a)
s = s[:a] + '''        j = candidates[np.argmax(score)]
        pair_ids = np.flatnonzero(np.any(prob.pairs == j, axis=1))
        k = pair_ids[0] if pair_ids.size else -1
        if k >= 0 and exclusive[k]:
            indices = prob.pairs[k]
            first = int(abs(w[indices[0]]) >= abs(w[indices[1]]))
            for bit in (first, 1-first):
                child = state.copy()
                child[indices] = (bit, 1-bit)
                visit(child, exclusive, w, lb)
        elif k >= 0 and np.all(state[prob.pairs[k]] == -1) and nodes+3 <= 600:
            indices = prob.pairs[k]
            # Keep interchangeable single-sensor choices in the same child
            # while separately bounding the zero- and two-sensor cases.
            group = exclusive.copy()
            group[k] = True
            visit(state, group, w, lb)
            for bit in (0, 1):
                child = state.copy()
                child[indices] = bit
                visit(child, exclusive, w, lb)
        else:
            first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
            for bit in (first, 1-first):
                child = state.copy()
                child[j] = bit
                visit(child, exclusive, w, lb)
''' + s[b:]
p.write_text(s)
