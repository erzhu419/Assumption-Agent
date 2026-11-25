import itertools, pickle, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import lsq_linear
from certified_l0_solver import solve_certified_box_l0


def loss(a, w):
    r = a['X'] @ w - a['y']
    return .5*(r@r + a['ridge']*(w@w)) + a['l0_penalty']*np.count_nonzero(np.abs(w)>1e-9)


def oracle(a):
    X,y=a['X'],a['y']; p=X.shape[1]; B=a['box']
    A=np.vstack((X,np.sqrt(a['ridge'])*np.eye(p))); b=np.r_[y,np.zeros(p)]
    opt=np.inf
    for mask in itertools.product((False,True),repeat=p):
        cap=np.where(mask,B,min(B,1e-9))
        w=lsq_linear(A,b,bounds=(-cap,cap),method='bvls',tol=1e-13).x
        opt=min(opt,loss(a,w))
    return opt


def check(spec):
    name,a,ref=spec; start=time.monotonic()
    try:
        if ref == 'oracle': ref=oracle(a)
        r=solve_certified_box_l0(a); w=np.array(r['weights']); obj=loss(a,w); lb=r['certified_lower_bound']; gap=obj-lb
        assert w.shape==(a['X'].shape[1],) and np.max(np.abs(w),initial=0)<=a['box']
        assert abs(r['objective']-obj)<=1e-10*(1+abs(obj)) and 0<=gap<=1e-5 and 1<=r['nodes_explored']<=600
        assert ref is None or (lb<=ref+1e-10*(1+abs(ref)) and obj<=ref+1e-5)
        return (name,'ok',r['nodes_explored'],gap,round(time.monotonic()-start,3))
    except Exception as e:
        return (name,'FAIL',type(e).__name__,str(e),round(time.monotonic()-start,3))


def main():
    specs=[]
    for i,a in enumerate(pickle.load(open('/holdout_task/public_box_l0_instances.pkl','rb'))): specs.append((f'public-{i}',a,None))
    def add(name,X,y,ridge,box,pen,ref=None):
        specs.append((name,dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=pen),ref))
    for seed in range(6):
        rng=np.random.default_rng(seed)
        X=rng.normal(size=(13,7)); X[:,1]=X[:,0]+.05*X[:,1]; y=2*X[:,0]-.6*X[:,3]+rng.normal(size=13)
        add(f'oracle-{seed}',X,y,(.0001,.15,1)[seed%3],(.15,.7,2)[seed%3],(.15,.6,1.2)[seed%3],'oracle')
    add('all-unpaid-box',np.array([[1e8,-3e8],[0,1e7]]),np.array([.3,-.01]),.1,1e-9,1.,'oracle')
    add('threshold-detectable',np.diag([1e8,1e8,1e8]),np.array([.08,.17,2.]),.5,.5,.02,'oracle')
    for seed in range(16):
        rng=np.random.default_rng(seed)
        for name,n,common,dup,ridge in [('dense',24,0,0,.35),('common',22,2,0,.15),('low-rank',8,0,0,.025),('neardup',24,0,.025,.05)]:
            X=rng.normal(size=(n,15))/np.sqrt(n)
            X+=common*rng.normal(size=(n,1))/np.sqrt(n)
            if dup: X[:,1:14:2]=X[:,:14:2]+dup*X[:,1:14:2]
            w=rng.normal(size=15); w[rng.random(15)<.5]=0
            y=X@w+rng.normal(size=n)*.25
            add(f'{name}-{seed}',X,y,ridge,(.45,1.5,3.)[seed%3],(.06,.18,.5,.85)[seed%4])
    p=15
    add('orthogonal-tie',np.eye(p),np.full(p,1.5),.125,2.,1.,None)
    H=np.repeat(np.eye(8),2,axis=1)[:,:15]
    add('duplicate-blocks',H,np.full(8,1.),.02,2.,.22)
    add('duplicate-coupled',np.vstack((H,np.full((1,15),.14))),np.r_[np.full(8,1.),.25],.02,2.,.22)
    with ProcessPoolExecutor(8) as pool:
        rows=list(pool.map(check,specs))
    for row in rows:
        if row[1]!='ok' or row[0].startswith(('public','oracle')) or row[2]>79 or row[0] in ('orthogonal-tie','duplicate-blocks','duplicate-coupled','all-unpaid-box','threshold-detectable'):
            print(*row)
    good=[r for r in rows if r[1]=='ok']; print('passed',len(good),'/',len(rows),'worst nodes',max(good,key=lambda r:r[2]),'worst gap',max(good,key=lambda r:r[3]))

if __name__=='__main__': main()
