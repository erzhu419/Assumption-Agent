import sys
sys.path.insert(0, '/tmp')
from test_proof_box import check
import numpy as np

def run(name, X, y, l2, l0, M, gt=None):
    print(name, check(dict(X=X, y=y, lam2=l2, lam0=l0, M=M),gt),flush=True)

p=22
v=np.linspace(-2,2,p); v[4]=0
for M in (.2, 1.1, 4.):
    for l0 in (.01,.22,10.):
        w=np.sign(v)*np.minimum(M,abs(v)/1.14)
        gt=.5*(v@v)+np.minimum(0,l0-abs(v*w)+.57*w*w).sum()
        run(('orthogonal',M,l0),np.eye(p),v,.14,l0,M,gt)
for seed in range(8):
    rng=np.random.default_rng(seed)
    p=19+seed%4
    X=rng.normal(size=(44,p)); X/=np.linalg.norm(X,axis=0)
    # A subset of columns shares a factor; some instances have exact duplicates.
    if seed>=3:
        X[:,1:8]=X[:,[0]] + (.12 if seed<6 else 0.)*X[:,1:8]
        X[:,1:8]/=np.linalg.norm(X[:,1:8],axis=0)
    coeff=np.zeros(p); coeff[rng.choice(p,6,replace=False)]=rng.choice([-1.,1.],6)*rng.uniform(.5,2.,6)
    y=X@coeff+rng.normal(scale=.12,size=len(X))
    run(('mixed',seed),X,y,.14,.22,(.35,1.1,2.)[seed%3])
