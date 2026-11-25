from pathlib import Path
p=Path('/holdout_task/certified_l0_solver.py')
s=Path('/tmp/l0_v07.py').read_text()
pos=s.index('        if bounds is not None and lb < prob.best-_GAP:')
s=s[:pos]+'''        if nodes == 1 and bounds is not None and lb < prob.best-_GAP:
            # Try to certify the zero- and two-activation cells directly. A
            # tangent optimized for the unrestricted cube can be weak on them.
            for k, indices in enumerate(prob.pairs):
                for cell in (0,3):
                    if not allowed[k,cell] or bounds[k,cell] >= prob.best-_GAP:
                        continue
                    nodes += 1  # Count each conditional bounding subproblem.
                    fixed_state = state.copy()
                    fixed_state[indices] = flags[cell]
                    restricted = allowed.copy()
                    restricted[k] = False
                    restricted[k,cell] = True
                    alt, wc, _ = prob.relaxation(fixed_state, prob.w, 256)
                    if alt < prob.best-_GAP:
                        alt = max(alt,prob.optimize_pairs(fixed_state,wc,180,restricted)[0])
                    bounds[k,cell] = max(bounds[k,cell],alt)
''' + s[pos:]
a=s.index('        # Split off the best-certified group')
b=s.index('    lb = min(terminal',a)
s=s[:a]+'''        possible = state == -1
        for k, indices in enumerate(prob.pairs):
            opts = flags[allowed[k]]
            possible[indices] = np.any(opts != opts[0], axis=0)
        candidates = np.flatnonzero(possible)
        if not candidates.size:
            visit(state,allowed,w,lb)
            continue
        a = np.abs(w[candidates]); sg = np.abs(g[candidates]); d = prob.d
        z0 = np.minimum(sg/d, prob.t); z1 = np.minimum(sg/d, prob.box)
        score = np.minimum(np.maximum(a-z0, 0)*d, np.maximum(z1-a, 0)*d)
        score += 1e-3*(np.abs(z1*(.5*d*z1-sg)+prob.lam-z0*(.5*d*z0-sg))<1e-7)*a
        j = candidates[np.argmax(score)]
        first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
        for bit in (first, 1-first):
            child = state.copy()
            child[j] = bit
            mask = allowed
            lower = lb
            if pair_members[j]:
                k, col = np.argwhere(prob.pairs == j)[0]
                mask = allowed.copy()
                mask[k] &= flags[:,col] == bit
                if bounds is not None:
                    lower = max(lower,float(np.min(bounds[k,mask[k]])))
            visit(child,mask,w,lower)
''' + s[b:]
p.write_text(s)
