import contextlib, io, runpy
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    check=runpy.run_path('/tmp/check_multicenter.py')
m,replay=check['m'],check['replay']
# Analytic lower bound on the number of leaves for the duplicate-pair instance.
l0,l2,M,y=.22,.14,1.1,.96
w=y/(1+l2)
chord=(w/M)*(l0+.5*l2*M*M)
loss=l0+.5*l2*w*w-chord
print('duplicate-pair: free optimum activation loses',loss,'; required leaves >=',2**11+1,flush=True)
assert 0<w<M and loss>1e-5
# Broaden full-size certificate testing on the public cycle structure and on
# dense designs with progressively poorer conditioning.
for kind in ('cycles','dense','correlated'):
    worst=[0,0,0.]; failures=[]
    for seed in range(36):
        rng=np.random.default_rng(2000+seed); p=22
        if kind=='cycles':
            G=np.eye(p)
            for grp in np.split(rng.permutation(p),2):
                for j,k in zip(grp,np.roll(grp,1)): G[j,k]=G[k,j]=rng.uniform(.19,.33)*rng.choice([-1,1])
            X=np.linalg.cholesky(G).T
            h=rng.uniform(.55,.84,p)*rng.choice([-1,1],p)
            target=np.linalg.solve(X.T,h)
        else:
            X=rng.normal(size=(44,p)); X+=({'dense':0.,'correlated':2.}[kind])*rng.normal(size=(44,1)); X/=np.linalg.norm(X,axis=0)
            target=X@rng.normal(0,.65,p)+rng.normal(0,.20,44)
        a=dict(X=X,y=target,lam0=l0,lam2=l2,M=M)
        try:
            s=m._sparse_fit(a); n,gap=replay(a,s)
        except (AssertionError,RuntimeError) as e:
            failures.append((seed,str(e))); continue
        if n>worst[0]: worst[:2]=[n,seed]
        worst[2]=max(worst[2],gap)
    print(kind,'worst leaves, seed, gap',worst,'failures',failures,flush=True)
