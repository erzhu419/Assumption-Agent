import importlib.util, pickle, sys, time
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/proof_box_l0.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def check(label, pr, truth=None):
    X,y,l0,l2,M=(pr[k] for k in ('X','y','lam0','lam2','M')); p=X.shape[1]; t=time.monotonic()
    try:
        ans=m.solve_proof_carrying_box_l0(pr)
        w=ans['weights']; L=ans['certified_lower_bound']; leaves=ans['subcube_leaves']; val=ans['objective']
        assert w.shape==(p,) and np.isfinite(w).all() and np.max(abs(w))<=M
        r=X@w-y; actual=.5*(r@r+l2*(w@w))+l0*np.count_nonzero(abs(w)>1e-8)
        assert abs(val-actual)<1e-10 and 0<=val-L<=1e-5 and 1<=len(leaves)<=2048
        if truth is not None: assert abs(actual-truth)<1e-5, (actual,truth)
        ints=1<<np.arange(p,dtype=np.int64); zs=[]; aa=[]; volume=0; G=X.T@X
        for leaf in leaves:
            z,a=leaf['fixed_zero'],leaf['fixed_active']; assert z.dtype==a.dtype==bool and z.shape==a.shape==(p,) and not np.any(z&a)
            d,c=leaf['minorant_diagonal'],leaf['minorant_center']; assert d.shape==c.shape==(p,) and np.all(d>=0) and np.linalg.eigvalsh(G-np.diag(d))[0]>=-1e-10
            r=y-X@c; g=X.T@r; h=abs(g+d*c); b=np.minimum(M,h/(l2+d)); on=l0-b*(h-.5*(l2+d)*b)
            witness=.5*(r@r+(d*c)@c)+g@c+on[a].sum()+np.minimum(0,on[~(z|a)]).sum()
            assert L<=leaf['lb']+1e-6 and leaf['lb']<=witness+1e-10
            Z=int(ints@z); A=int(ints@a)
            assert all((Z&av)|(A&zv) for zv,av in zip(zs,aa))
            zs.append(Z); aa.append(A); volume+=1<<(p-int(np.count_nonzero(z|a)))
        assert volume==1<<p
        print(label, 'gap',val-L,'leaves',len(leaves),'s',round(time.monotonic()-t,3),flush=True)
    except RuntimeError as e:
        print(label,'FAIL',str(e),'s',round(time.monotonic()-t,3),flush=True)

def make(X,y,l0=.14,l2=.12,M=1.): return dict(X=X,y=y,lam0=l0,lam2=l2,M=M)
for i, case in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))):
    pr = case['problem']; X, y = pr['X'], pr['y']; n, p = X.shape
    q = 22-p; l0, l2, M = (pr[k] for k in ('lam0','lam2','M'))
    # Pad the measured design to maximum dimension with independently solved
    # coordinates: exact activation ties and a box-saturated coordinate.
    v = np.full(q, np.sqrt(2*l0*(1+l2))); v[-1] = -2*M*(1+l2)
    b = np.clip(v/(1+l2), -M, M)
    extra = np.minimum(.5*v*v, .5*((v-b)**2+l2*b*b)+l0).sum()
    truth = case['gt']['exact_objective'] + extra
    A = np.zeros((n+q,22)); A[:n,:p] = X; A[n:,p:] = np.eye(q)
    u = np.r_[y,v]
    check(f'public-{i}-p22-ties-and-saturation', dict(pr,X=A,y=u), truth)
    # Orthogonal row sums/differences preserve the objective but destroy the
    # explicit row-disjoint structure between measured and appended columns.
    r = (np.arange(q)*7+2) % n; s = n+np.arange(q)
    xr, xs, yr, ys = A[r].copy(), A[s].copy(), u[r].copy(), u[s].copy()
    A[r], A[s] = .6*xr+.8*xs, -.8*xr+.6*xs
    u[r], u[s] = .6*yr+.8*ys, -.8*yr+.6*ys
    assert np.max(abs(A[r,p:])) > 0 and np.max(abs(A[r,:p])) > 0
    check(f'public-{i}-p22-row-sum-difference', dict(pr,X=A,y=u), truth)
