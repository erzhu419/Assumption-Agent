import importlib.util, pickle
import numpy as np
path = '/holdout_task/multicenter_certified_l0.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def replay(a, s):
    x,y,l0,l2,M = (a[k] for k in ('X','y','lam0','lam2','M'))
    w = s['weights']; p = x.shape[1]
    f = .5*np.linalg.norm(x@w-y)**2 + .5*l2*(w@w) + l0*np.count_nonzero(abs(w)>1e-8)
    assert np.max(np.abs(w)) <= M and abs(f-s['objective']) < 1e-11
    volume = 0; masks = []; bounds = []
    for leaf in s['subcube_leaves']:
        z,b,c,d = [leaf[k] for k in ('fixed_zero','fixed_active','minorant_center','minorant_diagonal')]
        assert z.dtype == b.dtype == bool and not np.any(z & b)
        mz,ma = (sum(1<<int(j) for j in np.flatnonzero(v)) for v in (z,b))
        assert all(mz&na or ma&nz for nz,na in masks)
        masks.append((mz,ma)); volume += 1 << (p - (z|b).sum())
        assert d.min() >= 0 and np.linalg.eigvalsh((x.T@x-np.diag(d))/max(1.,np.max(d)))[0] >= -1e-10
        r = y-x@c; g=x.T@r; linear=g+d*c; q=d+l2
        u=np.clip(linear/q,-M,M); v=np.clip(u,-1e-8,1e-8)
        on=l0-linear*u+.5*q*u*u; off=-linear*v+.5*q*v*v
        free=~(z|b)
        lb=.5*(r@r+(d*c)@c)+g@c+off[z].sum()+on[b].sum()+np.minimum(off[free],on[free]).sum()
        assert leaf['lb'] <= lb+1e-10
        bounds.append(leaf['lb'])
    assert volume == 1<<p and 1<=len(bounds)<=2048 and min(bounds)==s['certified_lower_bound']
    assert 0<=s['objective']-min(bounds)<=1e-5
    return len(bounds),s['objective']-min(bounds)

cases=pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl','rb'))
for i,case in enumerate(cases):
    a=case['study']['sparse_selection']
    for k in (1.,10.,100.,1000.,10000.,100000.,1000000.):
        b=dict(a,X=a['X']*k,lam2=a['lam2']*k*k,M=a['M']/k)
        try:
            s=m._sparse_fit(b)
            print(i,k,'replay',replay(b,s),'objective',s['objective'],flush=True)
        except (RuntimeError,AssertionError) as exc:
            print(i,k,'FAIL',type(exc).__name__,exc,flush=True)
