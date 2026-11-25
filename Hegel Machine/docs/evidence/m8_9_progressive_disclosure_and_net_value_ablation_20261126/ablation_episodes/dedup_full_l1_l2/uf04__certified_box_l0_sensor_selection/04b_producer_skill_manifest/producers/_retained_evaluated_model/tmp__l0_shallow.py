from pathlib import Path
p=Path('/holdout_task/certified_l0_solver.py')
s=Path('/tmp/l0_v09.py').read_text()
a=s.index('        if bounds is not None and lb < prob.best-_GAP:')
s=s[:a]+'''        if bounds is not None and lb < prob.best-_GAP and 1 <= depth <= 5 and nodes < 160:
            # Fixing a nearby indicator can make the double-activation cell
            # of another pair certifiable even when root screening could not.
            allowed = allowed.copy()
            for _ in range(3):
                np.maximum(bounds,lb,out=bounds)
                cut = allowed & (bounds >= prob.best-_GAP)
                if np.any(cut):
                    terminal = min(terminal,float(np.min(bounds[cut])))
                    allowed &= ~cut
                if np.any(~allowed.any(axis=1)):
                    return
                choices = [(bounds[k,c],k,c) for k in range(len(allowed))
                           for c in (0,3) if allowed[k,c] and allowed[k].sum()>2
                           and (k,c) not in prob.screened_here]
                if not choices or nodes >= 160:
                    break
                _,k,c = max(choices)
                prob.screened_here.add((k,c))
                restricted = allowed.copy()
                restricted[k] = False
                restricted[k,c] = True
                fixed_state = state.copy()
                for m,cols in enumerate(prob.pairs):
                    opts = flags[restricted[m]]
                    fixed = np.all(opts == opts[0],axis=0)
                    fixed_state[cols[fixed]] = opts[0,fixed]
                nodes += 1
                alt,wc,_ = prob.relaxation(fixed_state,w,160)
                if alt < prob.best-_GAP:
                    alt = max(alt,prob.optimize_pairs(fixed_state,wc,100,restricted)[0])
                bounds[k,c] = max(bounds[k,c],alt)
''' +s[a:]
s=s.replace('def visit(state, allowed, start, parent_lb):','def visit(state, allowed, start, parent_lb, depth=0):').replace('        nodes += 1\n        if allowed.size:', '        nodes += 1\n        screened_here = set()\n        if allowed.size:').replace('prob.screened_here','screened_here')
s=s.replace('state, allowed, bounds, w, g))','state, allowed, bounds, w, g, depth))').replace('state, allowed, bounds, w, g = heapq','state, allowed, bounds, w, g, depth = heapq').replace('visit(state,allowed,w,lb)','visit(state,allowed,w,lb,depth+1)').replace('visit(child,mask,w,lower)','visit(child,mask,w,lower,depth+1)')
p.write_text(s)
