import importlib.util, numpy as np, time
from scipy.optimize import lsq_linear
path='/holdout_task/coupled_interferometry_certified_l0.py'
spec=importlib.util.spec_from_file_location('eq02',path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def check(x,y,l2,l0,m,sol):
    p=x.shape[1]; w=np.array(sol['weights']); f=.5*np.sum((x@w-y)**2)+.5*l2*(w@w)+l0*np.count_nonzero(abs(w)>1e-8)
    assert np.max(abs(w))<=m and abs(f-sol['objective'])<1e-8
    ids=np.arange(1<<p,dtype=np.int64); seen=np.zeros(len(ids),dtype=np.uint8); bits=1<<np.arange(p); g=x.T@x
    slack=0
    for leaf in sol['subcube_leaves']:
        assert all(type(v) is bool for name in ['fixed_active','fixed_zero'] for v in leaf[name]); a=np.array(leaf['fixed_active']); z=np.array(leaf['fixed_zero']); assert not (a&z).any()
        seen[(ids & (bits@(a|z))) == (bits@a)]+=1
        c=np.array(leaf['minorant_center']); d=np.array(leaf['minorant_diagonal']); assert min(d)>=0 and np.linalg.eigvalsh(g-np.diag(d))[0]>=-1e-9
        r=y-x@c; h=x.T@r; v=h+d*c; k=d+l2
        u=np.clip(np.divide(v,k,out=m*np.sign(v),where=k>0),-m,m)
        on=l0+.5*k*u*u-v*u
        lb=.5*(r@r)+h@c+.5*np.dot(d,c*c)+on[a].sum()+np.minimum(on[~(a|z)],0).sum()
        slack=max(slack,leaf['lb']-lb)
    assert np.all(seen==1) and slack<=1e-9
    return f-sol['certified_lower_bound'],len(sol['subcube_leaves'])

def design(rng,n,p,case):
    x=rng.normal(size=(n,p))/np.sqrt(n)
    if case=='cluster':
        x=.48*x+.88*np.repeat(rng.normal(size=(n,(p+2)//3))/np.sqrt(n),3,axis=1)[:,:p]
    if case=='collinear':
        x=.17*x+.98*np.repeat(rng.normal(size=(n,(p+1)//2))/np.sqrt(n),2,axis=1)[:,:p]
    if case=='scaled': x*=np.geomspace(.15,4,p)
    if case=='bridge':
        x=np.zeros((n,p)); rr=np.arange(n-1); j=rng.integers(p,size=n-1); x[rr,j]=rng.normal(size=n-1); x[-1,:]=.07; x/=np.linalg.norm(x,axis=0)
    return x

for p in [18,19]:
    for i,case in enumerate(['dense','cluster','collinear','scaled','bridge']):
        rng=np.random.default_rng(31+p+i); x=design(rng,170,p,case); truth=np.zeros(p); truth[[1,5,8,12,16]]=[1.8,-.65,1.05,-.70,1.25]; y=x@truth+.08*rng.normal(size=len(x)); l2,l0,m=.025,.115,.84
        t=time.monotonic(); sol=mod._select(x,y,l2,l0,m); elapsed=time.monotonic()-t
        gap,count=check(x,y,l2,l0,m,sol); print(p,case,'seconds',round(elapsed,4),'gap',gap,'leaves',count,flush=True)

for seed in range(4):
    rng=np.random.default_rng(seed); n,p=50,7; x=design(rng,n,p,'cluster'); y=x@rng.normal(size=p)+.1*rng.normal(size=n); l2,l0,m=.025,.11,.67
    sol=mod._select(x,y,l2,l0,m); oracle=.5*y@y
    for mask in range(1,1<<p):
        a=((mask>>np.arange(p))&1).astype(bool); k=a.sum(); xx=np.vstack((x[:,a],np.sqrt(l2)*np.eye(k))); yy=np.r_[y,np.zeros(k)]
        fit=lsq_linear(xx,yy,bounds=(-m,m),tol=1e-13); oracle=min(oracle,fit.cost+l0*k)
    assert abs(sol['objective']-oracle)<1e-7, (seed,sol['objective'],oracle)
print('Exhaustive small box oracle: 4/4')
