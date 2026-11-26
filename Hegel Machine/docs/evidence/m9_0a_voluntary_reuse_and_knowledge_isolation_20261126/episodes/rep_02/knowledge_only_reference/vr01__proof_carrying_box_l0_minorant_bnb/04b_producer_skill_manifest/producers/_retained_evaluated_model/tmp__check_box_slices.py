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
for seed in range(30):
    rng=np.random.default_rng(seed+40)
    n=(12,22,44)[seed%3]; p=22
    X=rng.normal(size=(n,p)); X/=np.linalg.norm(X,axis=0)
    y=rng.normal(size=n)*(0.35,0.65,1.)[(seed//3)%3]
    check('dense-'+str(seed),make(X,y,l0=(.04,.12,.23)[(seed//9)%3],l2=(.01,.1)[seed%2],M=(.35,1.1)[(seed//2)%2]))
for seed in range(12):
    rng=np.random.default_rng(seed+80)
    p=22; A=np.eye(p)+.14*np.diag(rng.choice([-1.,1.],p-1),1)+.14*np.diag(rng.choice([-1.,1.],p-1),-1)
    A/=np.linalg.norm(A,axis=0)
    lam0=.14; lam2=.12
    g=np.sqrt(2*lam0*(1+lam2))*rng.choice([-1.,1.],p)*rng.uniform(.96,1.04,p)
    y=np.linalg.solve(A.T,g)
    check('coupled-near-tie-'+str(seed),make(A,y,l0=lam0,l2=lam2))
# A witness-independent lower bound on the number of leaves for 11 duplicate pairs.
a=.9/1.12
slope=.14+.5*.12
loss_of_bound=.14+.5*.12*a*a-slope*a
print('11-pair free-chosen-coordinate bound deficit:',loss_of_bound,'required leaves at least:',2**11+1)
assert loss_of_bound > 1.1e-5
