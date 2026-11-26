from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

# Reuse the full-grid fixtures and independent operator without rerunning baseline tests.
source = Path('/tmp/test_robin_eq01.py').read_text().split('\nwith threadpool_limits(limits=1, user_api=\'blas\'):\n    total =')[0]
exec(compile(source, '/tmp/test_robin_eq01.py', 'exec'))
original_cg, original_lu = mod.cg, mod.splu

def stress(st):
    solves, factors = [], []
    def cg(*args, **kwargs):
        u, info = original_cg(*args, **kwargs)
        solves.append(u)
        return u, info
    class Factor:
        def __init__(self, *args, **kwargs):
            self.factor = original_lu(*args, **kwargs)
            self.calls = []
        def solve(self, rhs):
            u = self.factor.solve(rhs)
            if self is not factors[0]:
                self.calls.append(u)
            return u
    def lu(*args, **kwargs):
        factor = Factor(*args, **kwargs)
        factors.append(factor)
        return factor
    mod.cg, mod.splu = cg, lu
    start = perf_counter()
    try:
        out = mod.solve_robin_continuation_tomography(st)
    finally:
        elapsed = perf_counter() - start
        mod.cg, mod.splu = original_cg, original_lu
    b = st['source_modes'].reshape(3,-1).T
    errors = []
    fallbacks = iter(factors[1:])
    for j,k in enumerate(np.argsort(st['thetas'])):
        a,v = reference(st, st['thetas'][k])
        g = mod._RobinOperator(st['kappa_base'].shape, [st['Lx'],st['Ly'],st['Lz']], st['gamma_robin'])
        t = st['thetas'][k]
        impl = g.assemble(st['kappa_base']+t*st['kappa_mod1']+t**2*st['kappa_mod2'], st['sigma_base']+t*st['sigma_mod'])
        for m in range(3):
            target = v*b[:,m]
            u = solves[3*j+m]
            if np.linalg.norm(g.volume*b[:,m]-impl@u)>1e-8*np.linalg.norm(g.volume*b[:,m]):
                f = next(fallbacks)
                u = f.calls[0] + f.calls[1]
            errors.append(np.linalg.norm(a@u-target)/np.linalg.norm(target))
    print(f'gamma={st["gamma_robin"]:.3g} time={elapsed:.3f}s state_mode_fallbacks={len(factors)-1} max_residual={max(errors):.3g}', flush=True)
    return elapsed, max(errors)

with threadpool_limits(limits=1, user_api='blas'):
    timings = []
    for i in range(2):
        st = make_study(1)
        st['gamma_robin'] = (.08, .04)[i]
        st['sigma_base'] *= .5
        st['sigma_mod'] *= .5
        if i:
            st['kappa_mod2'] = np.flip(st['kappa_mod2'], axis=0).copy()
        timings.append(stress(st))
    print('Combined weaker-wall runtime:', sum(t[0] for t in timings))
    assert max(t[1] for t in timings) <= 1e-7 and sum(t[0] for t in timings) <= 25
