import pickle
from pathlib import Path
import runpy

import numpy as np

n = runpy.run_path('/tmp/test_robin_eq01.py')
m, test, generate = (n[k] for k in ('m','test','generate'))
public = Path('/holdout_task/public_eq01_studies.pkl')
if public.is_file():
    def studies(obj):
        if isinstance(obj, dict):
            if 'kappa_base' in obj and 'Y_obs' in obj:
                yield obj
            else:
                for value in obj.values():
                    yield from studies(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                yield from studies(value)
    found = list(studies(pickle.loads(public.read_bytes())))
    assert found, 'Public file is present but its study layout was not recognized'
    for i, s in enumerate(found):
        test(s, f'public study {i}', True)
else:
    print('Public study still not mounted; audit uses constructed fixtures.', flush=True)

pair = 0.
for kind in (0, 1):
    s = generate(242+kind, (28,28,28), kind)
    x,y,z = np.indices((28,28,28))
    mask = (x+2*y+z)%9 < 4 if not kind else (x+y//4)%12 < 5
    kb = s['kappa_base']
    s['kappa_mod1'] = 0.001*kb
    s['kappa_mod2'] = np.where(mask, 15.*kb, 0.001*kb)
    s['thetas'] = s['thetas'][[5,0,8,2,7,3,6,4,1] if kind else [1,0,8,2,7,3,6,4,5]]
    # The row sum is the integrated reaction and wall loss, independent of interior faces.
    theta = s['thetas'][-1]
    h = np.array([s['Lx'], s['Ly'], s['Lz']])/kb.shape
    A, v = m._assembler(kb.shape, h, s['gamma_robin'])
    kappa = kb+theta*s['kappa_mod1']+theta**2*s['kappa_mod2']
    sigma = s['sigma_base']+theta*s['sigma_mod']
    g = s['gamma_robin']
    expected = v*sigma.copy()
    for axis in range(3):
        for side in (0,-1):
            k = np.take(kappa, side, axis=axis)
            np.moveaxis(expected, axis, 0)[side] += (v/h[axis])/(1/g+0.5*h[axis]/k)
    loss_error = np.max(np.abs(A(kappa,sigma)@np.ones(kb.size)-expected.ravel()))
    assert loss_error < 1e-10, loss_error
    print(f'case {kind}: max cell conservation defect {loss_error:.3g}', flush=True)
    pair += test(s, f'strongly nonuniform coefficient shift {kind}', True)
print(f'Shifted two-study runtime: {pair:.3f} s', flush=True)
assert pair <= 25.

# Near alignment with a rank-deficient nuisance basis is where subtracting
# unprojected Gram matrices can erase the one-mode denominator.
rng = np.random.default_rng(31)
Z = rng.normal(size=(900,8))
Z[:,-1] = Z[:,0]+2*Z[:,3]
S = Z[:,:3] + 1e-5*rng.normal(size=(900,3))
Y = Z @ rng.normal(size=(8,4)) + S @ rng.normal(size=(3,4)) + 1e-5*rng.normal(size=(900,4))
s = dict(Z_nuisance=Z,Y_obs=Y,nuisance_ridge=1e-10,source_ridge=1e-8)
a,w = m._profile(S,Z,Y,s['nuisance_ridge'],s['source_ridge'])
aa,ww = n['oracle_profile'](s,S)
ea,ew = n['error'](a,aa),n['error'](w,ww)
print(f'Near-confounded nuisance: amplitude L2={ea:.3g}, joint L2={ew:.3g}', flush=True)
assert ea <= 1e-5 and ew <= 0.03
