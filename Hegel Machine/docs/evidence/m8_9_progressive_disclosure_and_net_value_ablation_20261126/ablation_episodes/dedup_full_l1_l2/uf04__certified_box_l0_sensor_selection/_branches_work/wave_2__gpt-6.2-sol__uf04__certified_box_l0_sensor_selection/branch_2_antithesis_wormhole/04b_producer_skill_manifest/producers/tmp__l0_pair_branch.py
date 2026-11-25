from pathlib import Path
p = Path('/holdout_task/certified_l0_solver.py')
s = p.read_text()
s = s.replace("        if key in self.tried:\n            return False\n        self.tried.add(key)\n", "        if key in self.tried:\n            return self.tried[key]\n")
s = s.replace('self.tried = set()', 'self.tried = {}')
s = s.replace('        value = self.objective(w)\n', '        self.tried[key] = w\n        value = self.objective(w)\n')
s = s.replace('            return True\n        return False\n', '        return w\n')
s = s.replace('            prob.fit(state == 1)\n', '            start = prob.fit(state == 1)\n')
s = s.replace('            lb = max(lb, prob.pair_bound(state,w))', '            lb = max(lb, prob.pair_bound(state,w), prob.pair_bound(state,prob.w))')
s = s.replace('and nodes+2 <= 600:', 'and nodes+2 <= 600:')
start = s.index('        j = candidates[np.argmax(score)]')
end = s.index('    lb = min(terminal',start)
s = s[:start] + '''        j = candidates[np.argmax(score)]
        pairs = prob.pairs[np.all(state[prob.pairs] == -1, axis=1)]
        if len(pairs) and nodes+4 <= 600:
            # Resolve strongly correlated activations together, before opening
            # more pairs. The four children form a disjoint cover of indicators.
            priority = np.zeros(p)
            priority[candidates] = score
            indices = pairs[np.argmax(priority[pairs].sum(axis=1))]
            main = (np.abs(w[indices]) > (prob.t+prob.bridge)/2).astype(int)
            for flip in ((0,0),(1,0),(0,1),(1,1)):
                child = state.copy()
                child[indices] = main ^ flip
                visit(child, w, lb)
        else:
            first = int(abs(w[j]) > (prob.t+prob.bridge)/2)
            for bit in (first, 1-first):
                child = state.copy()
                child[j] = bit
                visit(child, w, lb)
''' + s[end:]
p.write_text(s)
