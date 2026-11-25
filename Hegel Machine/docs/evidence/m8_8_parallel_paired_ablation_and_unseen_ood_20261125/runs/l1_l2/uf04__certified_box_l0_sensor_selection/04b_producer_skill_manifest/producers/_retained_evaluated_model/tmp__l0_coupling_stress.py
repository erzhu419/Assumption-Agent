import concurrent.futures as cf
import sys
import numpy as np
sys.path.insert(0, '/tmp')
from stress_l0 import check, inst, oracle

def run(task):
    kind, seed = task
    rng = np.random.default_rng(seed)
    p = 6 if kind == 'oracle' else 15
    n = 24
    if kind == 'triples':
        X = np.repeat(rng.normal(size=(n, 5)), 3, axis=1) + .015*rng.normal(size=(n, p))
        y = X[:, ::3]@rng.uniform(.4, 1.5, 5) + .15*rng.normal(size=n)
    elif kind == 'common':
        X = rng.normal(size=(n, 1)) + .015*rng.normal(size=(n, p))
        y = X[:, 0]*(1.25 + .25*(seed%3)) + .12*rng.normal(size=n)
    elif kind == 'pairs' or kind == 'oracle':
        X = np.repeat(rng.normal(size=(n, (p+1)//2)), 2, axis=1)[:, :p] + .03*rng.normal(size=(n, p))
        X[:, 1::2] *= -1
        y = X[:, ::2]@rng.uniform(.15, 1.4, (p+1)//2) + .2*rng.normal(size=n)
    else:
        X = rng.normal(size=(n, p))
        y = rng.normal(size=n)
        X /= np.linalg.norm(X, axis=0)
    a = inst(X, y, (.025,.15,.35)[seed%3], (.45,.9,1.5)[seed%3], (.2,.5,.85)[seed%3])
    ref = oracle(a) if kind == 'oracle' else None
    result = check(task, a, ref)
    if result is None:
        import pickle
        with open('/tmp/l0_extra_%s_%d.pkl' % task, 'wb') as f:
            pickle.dump(a, f)
    return task, result

if __name__ == '__main__':
    tasks = [(kind, seed) for kind in ('oracle', 'triples', 'common', 'pairs', 'normalized') for seed in range(703, 706)]
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for f in cf.as_completed([pool.submit(run, task) for task in tasks]):
            print(*f.result(), flush=True)
