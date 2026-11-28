"""Panel-informed readout with training-only selection of the shared basis."""
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
from scipy.linalg import solve


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _profile(u, v, design, lo, hi):
    def cost(f):
        a = design(u, f)
        return np.mean((a @ np.linalg.lstsq(a, v, rcond=None)[0] - v)**2)
    grid = np.linspace(lo, hi, 121)
    j = int(np.argmin([cost(g) for g in grid]))
    f = minimize_scalar(cost, bounds=(grid[max(j-1, 0)], grid[min(j+1, len(grid)-1)]), method='bounded').x
    return f, np.linalg.lstsq(design(u, f), v, rcond=None)[0]


def _inter(x, v, triangular=False):
    if triangular:
        i, j = np.triu_indices(x.shape[1]); return x[:, i]*v[:, j]
    return (x[:, :, None]*v[:, None, :]).reshape(len(x), -1)


def _feature_bank(study, coords):
    q = np.vstack((coords['cubic_train'], coords['cubic_test']))
    x = np.vstack((study['train']['x'], study['test']['x']))
    z = np.cbrt(q)
    p = study['panel_b']; b_map = lambda x, f: _aff(np.cos(f*x))
    f, wb = _profile(p['u'], p['v'], b_map, .2, 9.)
    c = study['panel_c']; c_map = lambda x, g: _aff(np.arctan(g*x))
    g, wc = _profile(c['u'], c['v1'], c_map, .3, 6.)
    c2_map = lambda x, h: np.column_stack((_aff(x), np.cos(h*x)))
    h, wd = _profile(c['u'], c['v2'], c2_map, .3, 6.)
    cz, sz, cx = np.cos(f*z), np.sin(f*z), np.cos(f*x)
    b0 = b_map(x, f)@wb
    a = np.column_stack((_aff(z), q, cz, cx, np.tanh(b0), _inter(sz, sz, True)))
    yield 'baseline', a, (0.05,), False
    for nm, u in [('latent', z), ('raw', x), ('cubic', q)]:
        cb, sb = np.cos(f*u), np.sin(f*u)
        c1, c2 = c_map(u, g)@wc, c2_map(u, h)@wd
        bb = b_map(u, f)@wb
        base = np.column_stack((cb, cx, np.arctan(g*u)))
        for k, v in enumerate([base, np.c_[base,sb], np.c_[base,np.cos(g*u)], np.c_[base,c1,c2], np.c_[base,_inter(sb,sb,True)], np.c_[base,_inter(cb,cb,True)], np.c_[base,_inter(sb,np.sin(f*x))], np.c_[base,_inter(cb,c1)], np.c_[base,np.tanh(bb),_inter(sb,sb,True)], np.c_[base,_inter(np.sin(f*x),np.sin(f*x),True)], np.c_[base,u,u*u,u**3]]):
            yield nm+str(k), v, (.001,.025,.12,.5), True
        for vname, v in [('trig',np.c_[cb,sb]),('mix',np.c_[cb,cx,c1,u]),('panels',np.c_[bb,c1,c2])]:
            yield nm+vname, v, (.001,.015,.1), 'kernel'


def _fit(v, idx, y, ix, penalty, mode):
    if mode == 'kernel':
        a = v / np.maximum(v[idx].std(0), 1e-8)
        def kernel(a,b):
            e = np.exp(-.5*((a[:,None,:]-b[None,:,:])/4.)**2)
            return 1. + .85*e.mean(2)+.15*e.prod(2)
        k = kernel(a[idx],a[idx]); k.flat[::len(k)+1] += penalty
        return kernel(a[ix],a[idx]) @ solve(k,y[idx],assume_a='pos')
    if mode:
        a = (v-v[idx].mean(0))/np.maximum(v[idx].std(0),1e-8)
        ym = y[idx].mean(0); yy = y[idx]-ym
        k=a[idx].T@a[idx]; k.flat[::len(k)+1] += penalty*len(idx)
        return ym + a[ix]@solve(k,a[idx].T@yy,assume_a='pos')
    av=v[idx]; k=av.T@av; k.flat[::len(k)+1]+=penalty
    return v[ix]@solve(k,av.T@y[idx],assume_a='pos')


def run_node(inputs: dict) -> dict:
    study, coords = inputs['__input__'], inputs['node_00']
    tr = study['train']; y = np.column_stack((tr['y'],tr['traj']))
    n = len(y); idx = np.arange(n); test = np.arange(n,n+len(study['test']['x']))
    folds = np.array_split(np.random.default_rng(197).permutation(n),5)
    splits = [(np.setdiff1d(idx,j),j) for j in folds]
    models=[]
    for nm,v,ps,mode in _feature_bank(study,coords):
        for p in ps:
            e=np.array([np.abs(_fit(v,i,y,j,p,mode)-y[j]).mean(0) for i,j in splits]); e=np.c_[e[:,0],e[:,1:].mean(1)]
            score=.8*e.mean(0)+.2*e.max(0)
            models.append((score,nm,v,p,mode))
    out=[]
    for k in range(2):
        cand=sorted(models,key=lambda m:m[0][k]); selected=[]; seen=set()
        for score,nm,v,p,mode in cand:
            if nm in seen: continue
            seen.add(nm); selected.append((score[k],_fit(v,idx,y,test,p,mode)))
            if len(selected)>=4: break
        err=np.array([m[0] for m in selected]); weights=np.exp(-14*(err-err.min())/max(err.min(),.01)); weights/=weights.sum()
        out.append(np.einsum('i,ijk->jk',weights,np.array([m[1] for m in selected])))
    yh, state=out[0][:,0],out[1][:,1:]
    return {'y_hat': yh,'state_pred': state,'diagnostics_vec': np.array([yh.mean(),yh.std(),state[:,-1].mean(),state[:,-1].std()])}
