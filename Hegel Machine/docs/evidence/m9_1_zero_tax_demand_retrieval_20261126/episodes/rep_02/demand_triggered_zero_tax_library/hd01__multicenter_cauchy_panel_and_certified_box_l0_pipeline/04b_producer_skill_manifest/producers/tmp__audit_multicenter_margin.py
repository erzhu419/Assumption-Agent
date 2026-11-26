import pickle
import runpy
import time

import numpy as np

solve = runpy.run_path('/holdout_task/multicenter_certified_l0.py')['_sparse_fit']
items = pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl', 'rb'))
worst = (0., None)
largest = (0, None)
for i, item in enumerate(items):
    original = item['study']['sparse_selection']
    # Sweep penalties around support transitions while retaining the measured X and y.
    for M in (0.25, 0.55, original['M']):
        for lam0 in (0.03, 0.09, 0.16, 0.22, 0.30, 0.42, 0.58):
            spec = dict(original, M=M, lam0=lam0)
            t = time.monotonic()
            out = solve(spec)
            X, y, ridge = (spec[k] for k in ('X', 'y', 'lam2'))
            G, b = X.T @ X, X.T @ y
            w = out['weights']
            r = X @ w - y
            F = .5 * (r @ r + ridge * (w @ w)) + lam0 * np.count_nonzero(abs(w)>1e-8)
            assert np.all(abs(w) <= M) and abs(F-out['objective']) < 1e-10
            leaves = out['subcube_leaves']; p = len(w)
            masks = []; volume = 0
            for e in leaves:
                f, a = e['fixed_zero'], e['fixed_active']
                z, d = e['minorant_center'], e['minorant_diagonal']
                R = G-np.diag(d)
                assert np.min(d)>=0 and np.linalg.eigvalsh(R)[0]>=-1e-10
                g, h = R@z-b, d+ridge
                u = np.clip(-g/h, -M, M)
                v = np.clip(-g/h, -1e-8, 1e-8)
                active = .5*h*u*u+g*u+lam0
                inactive = .5*h*v*v+g*v
                L = .5*(y@y-z@R@z)+np.where(f,inactive,np.where(a,active,np.minimum(inactive,active))).sum()
                assert -1e-10 <= L-e['lb'] <= 1e-8
                fm = sum(1 << int(j) for j in np.flatnonzero(f)); am = sum(1 << int(j) for j in np.flatnonzero(a))
                assert not fm & am and all((fm&aa) or (am&ff) for ff,aa in masks)
                masks.append((fm,am)); volume += 1 << (p-(fm|am).bit_count())
            assert volume == 1 << p
            gap = F-out['certified_lower_bound']; label = (i, M, lam0)
            if gap > worst[0]: worst = gap, label
            if len(leaves) > largest[0]: largest = len(leaves), label
            if not (0 <= gap <= 1e-5 and 1 <= len(leaves) <= 2048):
                print('FAIL', label, gap, len(leaves), time.monotonic()-t, flush=True)
print('42 real-design penalty/box slices; worst gap:', worst, 'largest certificate:', largest)
assert worst[0] <= 1e-5 and largest[0] <= 2048
