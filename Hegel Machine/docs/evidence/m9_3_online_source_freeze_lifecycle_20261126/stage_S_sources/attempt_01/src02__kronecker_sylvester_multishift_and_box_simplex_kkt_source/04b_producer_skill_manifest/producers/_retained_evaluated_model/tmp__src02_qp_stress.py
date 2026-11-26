import importlib.util, pickle, sys
from collections import deque
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/kronecker_box_simplex_kkt_source_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_src02_studies.pkl', 'rb') as f:
    shape = pickle.load(f)[0]['study']['y_obs'].shape
rng = np.random.default_rng(293)
rng.normal(size=shape)

def trace(frame, event, arg):
    if frame.f_code is s._solve_capped.__code__:
        if event == 'line' and frame.f_lineno == frame.f_code.co_firstlineno + 10:
            v = frame.f_locals
            tail.append({k: (v[k].copy() if isinstance(v[k], np.ndarray) else v[k]) for k in ('bound','w','trial','hits','fractions','g','nu') if k in v})
        return trace
    return None

worst, feas = 0., 0.
for k in range(1200):
    n = 14; u = rng.uniform(.04,2)
    b = u * (int(rng.integers(0,n+1)) if k%2 else rng.uniform(0,n))
    d = rng.normal(size=(1+k%30,n))*10.**rng.uniform(-3,2)
    if k%4==0: d[:,1::2] = d[:,::2]
    y = rng.normal(size=len(d))*10.**rng.uniform(-3,2)
    if k%3==0:
        d = np.vstack((d, 10.**rng.uniform(-5,0)*np.eye(n)))
        y = np.r_[y,np.zeros(n)]
    try:
        w = s._solve_capped(d,y,u,b)
    except RuntimeError:
        print('failed slice',k,'box',u,'budget',b,'ratio',b/u,'design rank',np.linalg.matrix_rank(d),'shape',d.shape, flush=True)
        np.savez('/tmp/src02_failed_qp.npz', d=d,y=y,u=u,b=b)
        tail = deque(maxlen=5)
        np.set_printoptions(precision=17)
        sys.settrace(trace)
        try:
            s._solve_capped(d,y,u,b)
        finally:
            sys.settrace(None)
            for v in tail: print(v,flush=True)
    g = d.T@(d@w-y)
    gap = g@w - np.sort(g)@np.clip(b-u*np.arange(n),0,u)
    worst = max(worst,gap); feas = max(feas,abs(w.sum()-b))
    assert np.min(w)>=0 and np.max(w)<=u and abs(w.sum()-b)<1e-12 and abs(gap)<1e-6,(k,gap,w)
print('1200 QP slices: max gap',worst,'max budget error',feas)
