from check_proof_box import check,m
import numpy as np,time

def run(name,X,y,l0,l2,M):
    q=dict(X=X,y=y,lam0=l0,lam2=l2,M=M)
    t=time.monotonic()
    try:
        s=m.solve_proof_carrying_box_l0(q)
        print(name,check(q,s),round(time.monotonic()-t,3),flush=True)
    except RuntimeError as e:
        print(name,str(e),round(time.monotonic()-t,3),flush=True)

p=22
for k in (9,10,11):
    X=np.eye(p)[:,np.arange(p)-((np.arange(p)<2*k)&(np.arange(p)%2==1))]
    y=np.zeros(p); y[:2*k:2]=1
    run(('duplicate pairs',k),X,y,.15,.12,1.)
for seed in range(8):
    rng=np.random.default_rng(seed+100)
    for rho in (.47,.65):
        if rho<.5:
            G=np.eye(p)+rho*(np.eye(p,k=1)+np.eye(p,k=-1))
        else:
            a=rng.normal(size=(p,p)); a=a.T@a; s=np.sqrt(a.diagonal()); G=(1-rho)*np.eye(p)+rho*a/s[:,None]/s
        X=np.linalg.cholesky(G).T
        h=np.sqrt(2*.22*1.14)*rng.uniform(.85,1.15,p)*rng.choice([-1,1],p)
        run(('near activation ties',seed,rho),X,np.linalg.solve(X.T,h),.22,.14,1.1)
