import itertools
import pickle
import runpy
import time
import numpy as np

solve = runpy.run_path('/holdout_task/proof_box_l0.py')['solve_proof_carrying_box_l0']

def check(name, problem, expected=None):
    start = time.perf_counter()
    result = solve(problem)
    X, y, l0, l2, M = (problem[k] for k in ('X','y','lam0','lam2','M'))
    p = X.shape[1]
    w = result['weights']
    residual = y-X@w
    objective = 0.5*(residual@residual + l2*(w@w)) + l0*np.count_nonzero(abs(w)>1e-8)
    leaves = result['subcube_leaves']
    L = result['certified_lower_bound']
    assert w.shape == (p,) and np.all(abs(w)<=M) and np.all((w==0)|(abs(w)>1e-8))
    assert abs(result['objective']-objective)<1e-10
    assert 1<=len(leaves)<=2048, (name,len(leaves))
    assert 0<=objective-L<=1e-5, (name,objective-L)
    if expected is not None:
        assert abs(objective-expected)<=1e-6, (name,objective,expected)
    cubes = []
    volume = 0
    gram = X.T @ X
    for leaf in leaves:
        z, a = (np.asarray(leaf[k]) for k in ('fixed_zero','fixed_active'))
        assert z.shape==a.shape==(p,) and z.dtype==a.dtype==bool and not np.any(z&a)
        zb, ab = (sum(1<<int(j) for j in np.flatnonzero(q)) for q in (z,a))
        cubes.append((zb,ab))
        volume += 1 << (p-int(np.count_nonzero(z|a)))
        d,c = (leaf[k] for k in ('minorant_diagonal','minorant_center'))
        assert d.shape==c.shape==(p,) and np.all(d>=0)
        assert np.linalg.eigvalsh(gram-np.diag(d))[0]>=-1e-9
        fit = X@c
        r = y-fit
        q = abs(X.T@r+d*c)
        b = np.minimum(M,q/(l2+d))
        net = l0-q*b+0.5*(l2+d)*b*b
        witness = 0.5*(r@r)+r@fit+0.5*((d*c)@c)+net[a].sum()+np.minimum(net[~(a|z)],0).sum()
        assert L<=leaf['lb']+1e-6 and leaf['lb']<=witness+1e-9
    assert volume==1<<p and all(zi&aj or zj&ai for (zi,ai),(zj,aj) in itertools.combinations(cubes,2))
    print(f'{name}: leaves={len(leaves)} gap={objective-L:.3g} box={np.count_nonzero(abs(w)==M)} time={time.perf_counter()-start:.3f}s',flush=True)

for i,case in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))):
    check(f'public-{i}',case['problem'],case['gt']['exact_objective'])
for seed in range(12):
    rng=np.random.default_rng(seed)
    p=22
    family=seed%4
    if family==0:
        G=np.eye(p)+.26*(np.eye(p,k=1)+np.eye(p,k=-1))
        perm=rng.permutation(p)
        X=np.linalg.cholesky(G[np.ix_(perm,perm)]).T
        y=np.linalg.solve(X.T,rng.uniform(.57,.83,p)*rng.choice([-1,1],p))
    elif family==1:
        X=rng.normal(size=(40,p))/np.sqrt(40)
        y=rng.normal(size=40)*.8
    elif family==2:
        X=rng.normal(size=(15,p))/np.sqrt(15)
        y=rng.normal(size=15)
    else:
        X=np.eye(p)
        X[:,-1]=0
        y=rng.uniform(-2,2,p)
    M=(1.1,.30,.55,.27)[family]
    check(f'stress-{seed}',dict(X=X,y=y,lam0=.22,lam2=.14,M=M))
