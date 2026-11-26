import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/multicenter_certified_l0.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

def check(problem, proof, expected=None):
    x, y, m, l0, l2 = (problem[k] for k in ('X', 'y', 'M', 'lam0', 'lam2'))
    w = proof['weights']
    p = x.shape[1]
    objective = 0.5*np.linalg.norm(x@w-y)**2 + 0.5*l2*np.linalg.norm(w)**2 + l0*np.count_nonzero(np.abs(w)>1e-8)
    assert w.shape == (p,) and np.max(abs(w)) <= m
    assert abs(proof['objective']-objective) < 1e-10
    if expected is not None:
        assert abs(expected-objective) < 1e-5, (expected,objective)
    leaves = proof['subcube_leaves']
    assert 1 <= len(leaves) <= 2048
    gram = x.T@x
    masks = []
    cover = 0
    for leaf in leaves:
        z, a = leaf['fixed_zero'], leaf['fixed_active']
        assert all(0 <= j < p for j in z+a) and len(set(z+a)) == len(z+a)
        zm, am = (sum(1 << j for j in ix) for ix in (z,a))
        assert all((zm & olda) or (am & oldz) for oldz,olda in masks)
        masks.append((zm, am))
        cover += 1 << (p-len(z+a))
        d,c = leaf['minorant_diagonal'],leaf['minorant_center']
        assert d.shape == c.shape == (p,) and min(d) >= 0
        assert np.linalg.eigvalsh(gram-np.diag(d))[0] >= -1e-12
        # Minimize the independent tangent quadratic for each indicator state.
        residual = x@c-y
        grad = x.T@residual
        curvature = d+l2
        linear = grad-d*c
        minimizer = np.clip(-linear/curvature,-m,m)
        active = l0 + linear*minimizer + 0.5*curvature*minimizer**2
        free = np.ones(p,bool)
        free[z+a] = False
        inactive_minimizer = np.clip(-linear/curvature, -min(m, 1e-8), min(m, 1e-8))
        inactive = linear*inactive_minimizer + 0.5*curvature*inactive_minimizer**2
        reconstructed = (.5*(residual@residual + (d*c)@c)-grad@c
                         + sum(active[a]) + sum(inactive[z])
                         + sum(np.minimum(active[free], inactive[free])))
        assert leaf['lb'] <= reconstructed + 1e-10
    assert cover == 1<<p
    assert proof['certified_lower_bound'] == min(leaf['lb'] for leaf in leaves)
    gap = objective - proof['certified_lower_bound']
    assert 0 <= gap <= 1e-5
    return len(leaves),gap

public=pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))
for idx,entry in enumerate(public):
    s,gt = entry['study'],entry['gt']
    c=s['cohort']; ref=c['X_raw'].copy()
    for center in np.unique(c['center_id']):
        rows=c['center_id']==center
        ctrl=c['controls'][rows]; ctrl=ctrl[:,(np.isfinite(ctrl).sum(0)>=2)]
        ref[rows]=(ref[rows]-np.median(np.nanmean(ctrl,axis=0)))/np.median(np.nanstd(ctrl,axis=0,ddof=0))
    groups=np.stack([c['center_id'],c['stratum']],axis=1)
    for j,column in enumerate(ref.T):
        sums=[]; masses=[]
        for cell in np.unique(groups,axis=0):
            subset=column[(groups==cell).all(axis=1)]; good=np.isfinite(subset)
            sums.append(sum(subset[good])/len(subset)); masses.append(good.mean())
        column[np.isnan(column)]=sum(sums)/sum(masses)
    r=solver.solve_multicenter_certified_l0(s)
    np.testing.assert_allclose(r['aligned_biomarkers'],ref,rtol=0,atol=3e-14)
    err=abs(ref-gt['Z_true']); seen=np.isfinite(c['X_raw']); beta=gt['beta_potency']; rel=lambda b: np.linalg.norm(b-beta)/np.linalg.norm(beta)
    center_errors=[rel(solver._potency(ref[c['center_id']!=k], c['stratum'][c['center_id']!=k],c['y_potency'][c['center_id']!=k])) for k in np.unique(c['center_id'])]
    print(f'public {idx}: MAE full {err.mean():.8f}, observed {err[seen].mean():.8f}; slope error {rel(r["robust_potency_weights"]):.5f}; max leave-center-out slope error {max(center_errors):.5f}; leaves/gap {check(s["sparse_selection"],r["certified_selection"],gt["exact_l0_objective"])}',flush=True)

def trial(name,x,y,l0=.22,l2=.14,box=1.1,expected=None):
    s=dict(X=x,y=y,M=box,lam0=l0,lam2=l2); t=time.monotonic()
    try:
        proof=solver._certify(s)
        print(name,check(s,proof,expected),round(time.monotonic()-t,3),flush=True)
    except (RuntimeError,AssertionError) as err:
        print(name,type(err).__name__,err,round(time.monotonic()-t,3),flush=True)

rng=np.random.default_rng(713)
for seed in range(12):
    p=22; n=48; u=np.linalg.qr(rng.normal(size=(n,p)))[0]; g=np.eye(p); ix=np.arange(p-1); g[ix,ix+1]=g[ix+1,ix]=rng.uniform(-.33,.33,p-1); x=u@np.linalg.cholesky(g).T
    y=x@rng.uniform(-1.15,1.15,p)+rng.normal(0,.15,n)
    trial('coupled band '+str(seed),x,y,box=[.55,1.1,1.35][seed%3])
for seed in range(4):
    x=rng.normal(size=(44,22)); x/=np.linalg.norm(x,axis=0); y=x@rng.uniform(-1.2,1.2,22)+rng.normal(0,.15,44)
    trial('dense '+str(seed),x,y)
p=22
x=np.eye(p); a=.14; l0=.22; y=np.full(p,np.sqrt(2*l0*(1+a)))
trial('orthogonal activation ties',x,y,expected=.5*y@y)
x=np.repeat(np.eye(p//2),2,axis=1); y=np.full(p//2,.84)
expected=(p//2)*min(.5*.84**2,.5*.84**2-.5*.84**2/(1+a)+l0)
trial('duplicate-pair ties',x,y,expected=expected)
