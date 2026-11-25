import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from certified_l0_solver import solve_certified_box_l0
rng = np.random.default_rng(820)
Q = np.linalg.qr(rng.normal(size=(32, 15)))[0]
failures = []
for case in ('orthogonal', 'pairs', 'weakly_coupled_pairs'):
    for ridge, fee in ((.12,.55),(.35,.85)):
        M = 2.
        if case == 'orthogonal':
            X = Q
            y = Q @ np.full(15,np.sqrt(2*fee*(1+ridge)))
        else:
            X = Q[:,np.arange(15)//2].copy()
            y = Q[:,:8] @ np.full(8,np.sqrt(2*fee*(1+ridge))+.12)
            if case == 'weakly_coupled_pairs':
                X += 1e-4*rng.normal(size=X.shape)
        try:
            out = solve_certified_box_l0(dict(X=X,y=y,ridge=ridge,box=M,l0_penalty=fee))
            assert 1 <= out['nodes_explored'] <= 600
            assert 0 <= out['objective'] - out['certified_lower_bound'] <= 1e-5
            print(case, ridge, 'nodes',out['nodes_explored'],'gap',out['objective']-out['certified_lower_bound'],flush=True)
        except RuntimeError as e:
            print(case,ridge,e,flush=True)
            failures.append((case, ridge))

assert not failures, failures
