import importlib.util, pickle, time
import numpy as np
path='/holdout_task/multicenter_certified_l0.py'
spec=importlib.util.spec_from_file_location('solver',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def replay(a,s):
    X,y,l0,l2,M=(a[k] for k in ('X','y','lam0','lam2','M'))
    p=X.shape[1]; G=X.T@X
    w=s['weights']; r=X@w-y
    F=.5*(r@r+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
    assert w.shape==(p,) and np.max(abs(w))<=M and abs(F-s['objective'])<=1e-12
    leaves=s['subcube_leaves']; assert 1<=len(leaves)<=2048
    volume=0; bounds=[]; masks=[]
    for leaf in leaves:
        z,a=(np.asarray(leaf[k]) for k in ('fixed_zero','fixed_active'))
        assert z.shape==a.shape==(p,) and z.dtype==a.dtype==bool and not (z&a).any()
        zz=sum(1<<int(j) for j in np.flatnonzero(z)); aa=sum(1<<int(j) for j in np.flatnonzero(a))
        assert all(zz&ap or aa&zp for zp,ap in masks)
        masks.append((zz,aa)); volume+=1<<(p-int((z|a).sum()))
        d,c=(np.asarray(leaf[k]) for k in ('minorant_diagonal','minorant_center'))
        assert np.min(d)>=0 and np.linalg.eigvalsh(G-np.diag(d))[0]>=-1e-11
        R=y-X@c; g=X.T@R; b=g+d*c
        u=np.clip(b/(d+l2),-M,M); v=l0-b*u+.5*(d+l2)*u*u
        lb=.5*(R@R + c@(d*c))+g@c+v[a].sum()+np.minimum(0,v[~(z|a)]).sum()
        assert leaf['lb']<=lb+1e-10 and leaf['lb']>=F-1e-5
        bounds.append(leaf['lb'])
    assert volume==1<<p and s['certified_lower_bound']==min(bounds)
    gap=F-min(bounds); assert 0<=gap<=1e-5
    return len(leaves),gap

for i,case in enumerate(pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))):
    study=case['study']; gt=case['gt']; out=m.solve_multicenter_certified_l0(study)
    c=study['cohort']; expected=c['X_raw'].copy(); g=c['center_id']; t=c['stratum']
    for k in np.unique(g):
        rows=g==k; panel=c['controls'][rows]; cols=[col[np.isfinite(col)] for col in panel.T if np.isfinite(col).sum()>=2]
        expected[rows]=(expected[rows]-np.median([v.mean() for v in cols]))/np.median([v.std(ddof=0) for v in cols])
    wt=np.array([1/np.count_nonzero((g==u)&(t==v)) for u,v in zip(g,t)])
    for col in expected.T:
        ok=np.isfinite(col); col[~ok]=np.average(col[ok],weights=wt[ok])
    assert np.allclose(out['aligned_biomarkers'],expected,atol=2e-14,rtol=0)
    mae=np.mean(abs(expected-gt['Z_true'])); err=np.linalg.norm(out['robust_potency_weights']-gt['beta_potency'])/np.linalg.norm(gt['beta_potency']); assert err<=.08
    s=out['certified_selection']; assert abs(s['objective']-gt['exact_l0_objective'])<=1e-5
    print('PUBLIC',i,'exact recipe MAE',mae,'potency',err,'certificate',replay(study['sparse_selection'],s),flush=True)

worst={}; fail={}
for kind in ('signed_cycles','near_tie_cycles','dense','orthogonal_tie','duplicate_pairs','box_saturated'):
    for seed in range(12 if kind in ('signed_cycles','near_tie_cycles','dense') else 1):
        rng=np.random.default_rng(seed); p=22; lam0=.22; lam2=.14; box=1.1
        G=np.eye(p); target=rng.normal(0,.52,p)
        if kind.endswith('cycles'):
            for group in np.array_split(rng.permutation(p),2):
                for u,v in zip(group,np.roll(group,-1)):
                    G[u,v]=G[v,u]=rng.choice([-1,1])*.26
            if kind=='near_tie_cycles': target[:]=rng.choice([-1,1],p)*rng.uniform(.59,.78,p)
            X=np.linalg.cholesky(G).T; y=np.linalg.solve(X.T,target)
        elif kind=='dense':
            X=rng.normal(size=(44,p)); X/=np.linalg.norm(X,axis=0); y=X@rng.normal(0,.6,p)+rng.normal(0,.25,44)
        elif kind=='duplicate_pairs':
            X=np.repeat(np.eye(p//2),2,axis=1); y=np.full(p//2,.96)
        elif kind=='orthogonal_tie': X=np.eye(p); y=np.full(p,np.sqrt(2*lam0*(1+lam2)))
        else: X=np.eye(p); y=np.linspace(-9,9,p); box=.28
        a={'X':X,'y':y,'lam0':lam0,'lam2':lam2,'M':box}
        try:
            stamp=time.monotonic(); s=m._sparse_fit(a); leaves,gap=replay(a,s)
            if kind=='duplicate_pairs': assert abs(s['objective']-(p//2)*(.5*.96**2+lam0-.5*.96**2/(1+lam2)))<1e-8
            worst[kind]=tuple(max(i,j) for i,j in zip(worst.get(kind,(0,0,0)),(leaves,gap,time.monotonic()-stamp)))
        except (RuntimeError,AssertionError) as e:
            fail.setdefault(kind,[]).append((seed,str(e))); print('FAIL',kind,seed,str(e),flush=True)
            if len(fail[kind])>=2: break
print('STRESS worst (leaves,gap,seconds):',worst,'failures:',fail,flush=True)
