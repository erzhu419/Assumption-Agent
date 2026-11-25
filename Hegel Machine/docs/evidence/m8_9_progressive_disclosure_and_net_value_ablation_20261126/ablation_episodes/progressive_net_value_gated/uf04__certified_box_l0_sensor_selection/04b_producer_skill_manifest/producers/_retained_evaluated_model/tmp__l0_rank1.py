import importlib.util
import numpy as np
import time
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/certified_l0_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for seed in (81,82,83):
 rng = np.random.default_rng(seed)
 x = rng.normal(size=24); x /= np.linalg.norm(x)
 signs = np.where(np.arange(15) % 2, -1., 1.)
 noise = rng.normal(size=(24, 15)) / np.sqrt(24)
 for eps in (0., .0001, .001, .02, .08):
    X = (x[:, None] + eps*noise)*signs; y = 3.1*x
    t=time.monotonic(); r = m.solve_certified_box_l0(dict(X=X, y=y, ridge=.015, box=.6, l0_penalty=.15))
    w=np.array(r['weights']); e=X@w-y
    obj=.5*(e@e+.015*(w@w))+.15*np.count_nonzero(abs(w)>1e-9)
    assert r['objective']==obj and max(abs(w))<=.6 and r['nodes_explored']<=600
    print(seed, eps, 'gap',obj-r['certified_lower_bound'],'nodes',r['nodes_explored'],'secs',round(time.monotonic()-t,2), flush=True)
