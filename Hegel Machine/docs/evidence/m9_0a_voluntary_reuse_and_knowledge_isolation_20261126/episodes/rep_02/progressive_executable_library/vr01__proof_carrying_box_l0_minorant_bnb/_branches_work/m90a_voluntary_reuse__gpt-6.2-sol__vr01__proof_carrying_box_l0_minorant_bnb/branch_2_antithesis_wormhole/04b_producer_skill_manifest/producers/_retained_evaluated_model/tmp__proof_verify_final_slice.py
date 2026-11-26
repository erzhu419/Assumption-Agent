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

