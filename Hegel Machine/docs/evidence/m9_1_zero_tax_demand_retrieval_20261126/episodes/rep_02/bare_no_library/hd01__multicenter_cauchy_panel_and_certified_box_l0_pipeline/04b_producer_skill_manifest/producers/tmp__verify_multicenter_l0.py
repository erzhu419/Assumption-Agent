import importlib.util
import pickle
import time
import numpy as np

path = '/holdout_task/multicenter_certified_l0.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def verify(s, out, expected=None):
    X, y, M, l0, l2 = (s[k] for k in ('X','y','M','lam0','lam2'))
    p = X.shape[1]
    w = out['weights']
    assert w.shape == (p,) and np.max(np.abs(w)) <= M + 1e-12
    f = float(0.5*np.linalg.norm(X@w-y)**2 + 0.5*l2*np.linalg.norm(w)**2 + l0*np.count_nonzero(np.abs(w)>1e-8))
    assert abs(f-out['objective']) < 1e-9
    if expected is not None:
        assert abs(f-expected) < 1e-5, (f, expected)
    G = X.T @ X
    masks = []
    mass = 0
    lbs = []
    for leaf in out['subcube_leaves']:
        z, a = leaf['fixed_zero'], leaf['fixed_active']
        assert len(z+a) == len(set(z+a)) and all(0 <= j < p for j in z+a)
        zero, active = (sum(1 << j for j in js) for js in (z,a))
        assert all((zero & aa) or (active & zz) for zz,aa in masks)
        masks.append((zero,active))
        mass += 1 << (p-len(z)-len(a))
        d, c = leaf['minorant_diagonal'], leaf['minorant_center']
        assert d.shape == c.shape == (p,) and np.min(d) >= 0
        B = G-np.diag(d)
        assert np.linalg.eigvalsh(B)[0] >= -1e-10
        v = B@c-X.T@y
        ref = 0.5*(y@y-c@B@c)
        for j in range(p):
            curv = l2+d[j]
            t = np.clip(-v[j]/curv,-M,M)
            act = 0.5*curv*t*t+v[j]*t+l0
            t = np.clip(-v[j]/curv,-min(M,1e-8),min(M,1e-8))
            off = 0.5*curv*t*t+v[j]*t
            ref += off if j in z else act if j in a else min(off,act)
        assert 0 <= ref-leaf['lb']+1e-10 and ref-leaf['lb'] < 1e-8
        lbs.append(leaf['lb'])
    assert mass == 1 << p
    L = out['certified_lower_bound']
    assert abs(L-min(lbs)) < 1e-9
    gap = f-L
    assert -1e-12 <= gap <= 1e-5, gap
    print('objective',f,'gap',gap,'leaves',len(masks),'saturated',np.count_nonzero(np.isclose(np.abs(w),M)),flush=True)
    assert 1 <= len(masks) <= 2048, len(masks)

rows = pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))
for i,r in enumerate(rows):
    s,g = r['study'],r['gt']
    result = m.solve_multicenter_certified_l0(s)
    z,b = result['aligned_biomarkers'],result['robust_potency_weights']
    assert z.shape == g['Z_true'].shape and b.shape == g['beta_potency'].shape
    assert np.isfinite(z).all() and np.isfinite(b).all()
    assert np.linalg.norm(b-g['beta_potency'])/np.linalg.norm(g['beta_potency']) <= .08
    print('public',i,'alignment MAE',np.mean(np.abs(z-g['Z_true'])),flush=True)
    verify(s['sparse_selection'],result['certified_selection'],g['exact_l0_objective'])

rng = np.random.default_rng(523)
base = rows[1]['study']['sparse_selection']
X0 = base['X']
Q,_ = np.linalg.qr(np.column_stack((X0,rng.normal(size=(len(X0),2)))))
X = np.column_stack((X0,Q[:,-2:]))
for name,s in [
    ('p22-near-threshold-extra-columns',dict(base,X=X,y=base['y']+0.71*Q[:,-2]-0.705*Q[:,-1])),
    ('p22-box-saturated',dict(base,X=X,y=2.8*base['y']+1.8*Q[:,-1],M=.42)),
    ('p22-rank-deficient',dict(base,X=np.column_stack((X0,X0[:,0],X0[:,4])))),
    ('p22-correlated-dense-boundary',dict(base,X=X,y=X@np.linalg.solve(X.T@X,0.71*rng.choice([-1.,1.],22))))]:
    print(name,flush=True)
    start=time.monotonic()
    out=m._select(s)
    print('solve seconds',time.monotonic()-start,flush=True)
    verify(s,out)
