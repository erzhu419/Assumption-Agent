import itertools, pickle, runpy, time
import numpy as np
from scipy.optimize import lsq_linear

mod = runpy.run_path('/holdout_task/multicenter_certified_l0.py')

def verify(spec, res):
    X, y, lam2, lam0, M = [spec[k] for k in ('X', 'y', 'lam2', 'lam0', 'M')]
    G = X.T @ X
    p = len(G)
    w = res['weights']
    F = lambda u: .5*np.linalg.norm(X@u-y)**2+.5*lam2*np.linalg.norm(u)**2+lam0*np.count_nonzero(abs(u)>1e-8)
    assert w.shape == (p,) and np.all(abs(w)<=M) and abs(F(w)-res['objective'])<1e-10
    leaves = res['subcube_leaves']; assert 1 <= len(leaves) <= 2048
    volume = 0; masks = []; gaps = []
    for e in leaves:
        f, a = e['fixed_zero'], e['fixed_active']; assert f.shape == a.shape == (p,) and not np.any(f & a)
        z, d = e['minorant_center'], e['minorant_diagonal']; R = G-np.diag(d)
        assert np.min(d)>=0 and np.min(np.linalg.eigvalsh(R))>=-1e-10
        r = X@z-y; g = X.T@r-d*z; u = np.clip(-g/(d+lam2), -M, M)
        A = g*u+.5*(d+lam2)*u*u+lam0
        L = .5*(y@y-z@R@z)+np.sum(np.where(f,0,np.where(a,A,np.minimum(0,A))))
        delta = L-e['lb']; assert -1e-10 <= delta <= 1e-6, delta
        gaps.append(delta)
        fm = sum(1<<int(j) for j in np.flatnonzero(f)); am = sum(1<<int(j) for j in np.flatnonzero(a))
        assert all((fm&aa) or (am&ff) for ff,aa in masks)
        masks.append((fm,am)); volume += 1 << (p-(fm|am).bit_count())
    assert volume == 1<<p
    L = res['certified_lower_bound']; gap = F(w)-L
    assert -1e-10<=gap<=1e-5 and abs(L-min(F(w),min(e['lb'] for e in leaves)))<1e-10
    return len(leaves), gap, np.sum(abs(w)==M), max(gaps)

items = pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))
for i,item in enumerate(items):
    study, gt = item['study'], item['gt']; o = mod['solve_multicenter_certified_l0'](study); s = o['certified_selection']
    print('public',i,verify(study['sparse_selection'],s)); assert abs(s['objective']-gt['exact_l0_objective'])<1e-8
    beta = gt['beta_potency']; assert np.linalg.norm(o['robust_potency_weights']-beta)/np.linalg.norm(beta)<=.08

for seed in range(12):
    rng = np.random.default_rng(seed); p = 19+seed%4; H = np.eye(p)
    for j in range(p-1): H[j,j+1]=H[j+1,j]=rng.choice([-1,1])*(.26 if seed%2 else .45)
    if seed%3==0: H=.78*H+.22*np.ones((p,p))
    X=np.linalg.cholesky(H).T; w=np.where(rng.random(p)<.45,rng.choice([-1.,1.],size=p)*rng.uniform(.4,1.55,p),0.)
    spec=dict(X=X,y=X@w+rng.normal(0,.08,p),lam2=.14,lam0=.22,M=.43 if seed%3==1 else 1.1)
    t=time.monotonic(); o=mod['_sparse_fit'](spec); print('stress',seed,verify(spec,o),'seconds',round(time.monotonic()-t,3),flush=True)

for seed in range(3):
    rng = np.random.default_rng(seed); X=rng.normal(size=(14,8))/np.sqrt(14); y=rng.normal(size=14); M=.4; lam0=.22; lam2=.14
    spec=dict(X=X,y=y,M=M,lam0=lam0,lam2=lam2)
    o=mod['_sparse_fit'](spec); verify(spec,o)
    A=np.vstack((X,np.sqrt(lam2)*np.eye(8))); b=np.r_[y,np.zeros(8)]; exact=.5*(y@y)
    for k in range(1,9):
        for ids in itertools.combinations(range(8),k):
            rr=lsq_linear(A[:, ids],b,bounds=(-M,M),tol=1e-12); exact=min(exact,rr.cost+k*lam0)
    assert abs(exact-o['objective'])<1e-8
    print('small exhaustive',seed,abs(exact-o['objective']))
