import importlib.util, pickle, time
import numpy as np

spec = importlib.util.spec_from_file_location('proof_box_l0', '/holdout_task/proof_box_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check(name, q, exact):
    X,y,l0,l2,M = (q[k] for k in ('X','y','lam0','lam2','M'))
    p = X.shape[1]
    start = time.monotonic()
    out = mod.solve_proof_carrying_box_l0(q)
    w,L,leaves = (out[k] for k in ('weights','certified_lower_bound','subcube_leaves'))
    assert w.shape == (p,) and np.all(np.abs(w) <= M+1e-12)
    F = .5*np.sum((X@w-y)**2) + .5*l2*(w@w) + l0*np.count_nonzero(np.abs(w)>1e-8)
    assert abs(F-out['objective']) < 1e-10
    assert 0 <= F-L <= 1e-5 and abs(F-exact) <= 1e-5 and L <= exact+1e-9, (name,F,L,exact)
    assert 1 <= len(leaves) <= 2048
    masks, volume, bounds = [], 0, []
    G = X.T@X
    psd = np.inf
    for leaf in leaves:
        z,a = (np.asarray(leaf[k]) for k in ('fixed_zero','fixed_active'))
        assert z.shape == a.shape == (p,) and z.dtype == a.dtype == bool and not np.any(z&a)
        zi,ai = (sum(1<<int(i) for i in np.flatnonzero(m)) for m in (z,a))
        assert all((zi&ak) or (ai&zk) for zk,ak in masks), name
        masks.append((zi,ai))
        u = ~(z|a)
        volume += 1 << int(u.sum())
        if 'minorant_diagonal' in leaf:
            d,c = (np.asarray(leaf[k]) for k in ('minorant_diagonal','minorant_center'))
            assert d.shape == c.shape == (p,) and np.all(d >= 0)
            eig = np.linalg.eigvalsh(G-np.diag(d))[0]
            psd = min(psd,eig)
            assert eig >= -1e-10
            r = y-X@c
            const = .5*(r@r) + r@(X@c) + .5*((d*c)@c)
            h, curvature = np.abs(X.T@r+d*c), l2+d
        else:
            r = np.asarray(leaf['dual_nu'])
            assert r.shape == y.shape
            const = r@y-.5*(r@r)
            h, curvature = np.abs(X.T@r), l2
        b = np.minimum(M,h/curvature)
        cost = l0-h*b+.5*curvature*b*b
        witness = const+cost[a].sum()+np.minimum(cost[u],0).sum()
        assert np.isfinite(witness) and leaf['lb'] <= witness+1e-9, (name,leaf['lb'],witness)
        bounds.append(leaf['lb'])
    assert volume == 1<<p and L <= min(bounds)+1e-6
    print(name,'error',format(F-exact,'.3g'),'gap',format(F-L,'.3g'),'leaves',len(leaves),'PSD margin',format(psd,'.3g'),'sec',format(time.monotonic()-start,'.3g'),flush=True)

for i,c in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))):
    check(f'public-{i}',c['problem'],c['gt']['exact_objective'])

p=22
for M in (0.3, 2.):
    l0,l2=.22,.14
    b=min(M,np.sqrt(2*l0/(1+l2)))
    y=np.full(p,l0/b+.5*(1+l2)*b)*np.where(np.arange(p)%2,1,-1)
    check(f'orthogonal-tie-M{M}',dict(X=np.eye(p),y=y,lam0=l0,lam2=l2,M=M),.5*(y@y))

# Path Gram matrices: independently optimize every contiguous active segment,
# then use a path DP to obtain the exact minimum over all binary supports.
from scipy.optimize import lsq_linear
for rho in (.26,.48):
    for seed in (3,15):
        rng=np.random.default_rng(seed)
        l0,l2,M=.22,.14,.75
        G=np.eye(p)+rho*(np.eye(p,k=1)+np.eye(p,k=-1))
        h=rng.uniform(.38,1.28,p)*rng.choice([-1,1],p)
        X=np.linalg.cholesky(G).T
        y=np.linalg.solve(X.T,h)
        H=G+l2*np.eye(p)
        dp=np.zeros(p+1)
        for j in range(1,p+1):
            dp[j]=dp[j-1]
            for i in range(j):
                R=np.linalg.cholesky(H[i:j,i:j]).T
                b=np.linalg.solve(R.T,h[i:j])
                v=lsq_linear(R,b,bounds=(-M,M),tol=1e-13).x
                value=.5*np.dot(R@v,R@v)-v@h[i:j]+l0*(j-i)
                dp[j]=min(dp[j],dp[max(0,i-1)]+value)
        perm=rng.permutation(p)
        check(f'path-rho{rho}-seed{seed}',dict(X=X[:,perm],y=y,lam0=l0,lam2=l2,M=M),.5*(y@y)+dp[p])
