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
for i,e in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))): check('public'+str(i),e['problem'],e['gt']['exact_objective'])
p=22; l0=.14; l2=.12
for label,y in [('orthogonal-tie',np.full(p,np.sqrt(2*l0*(1+l2)))),('orthogonal-saturation',np.linspace(-4,4,p)),('zero-target',np.zeros(p))]:
    w=np.clip(y/(1+l2),-1,1); truth=np.minimum(.5*y*y,.5*((w-y)**2+l2*w*w)+l0).sum(); check(label,make(np.eye(p),y),truth)
for seed in range(12):
    rng=np.random.default_rng(seed); n=(14,22,44)[seed%3]; X=rng.normal(size=(n,p)); X/=np.linalg.norm(X,axis=0); y=rng.normal(size=n)*.75
    check('coupled-'+str(seed),make(X,y,l0=(.07,.14,.25)[(seed//3)%3],l2=(.03,.12)[seed%2],M=(.5,1.)[seed%2]))
for q in (4,7,9,10,11):
    X=np.pad(np.repeat(np.eye(q),2,axis=1),((0,0),(0,22-2*q))); y=np.full(q,.9)
    fact=np.array([.5*.9**2,l0+.5*l2*.9**2/(1+l2),2*l0+.5*l2*.9**2/(2+l2)])
    check('duplicate-pairs-'+str(q),make(X,y),q*fact.min())
