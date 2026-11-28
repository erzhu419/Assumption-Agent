import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
print('study fields', sorted(s), 'train fields', sorted(s['train']))
p = s['panel_c']
u = p['u']
for k in ('v1','v2'):
    v = p[k]
    scores = []
    for label, lo, hi, fn in [('candidate',.3 if k=='v1' else 1.8,1.6 if k=='v1' else 3.,lambda u,w:m._panel_basis(u,w,k)),('rational',.03,20.,lambda u,w:np.c_[np.ones(len(u)),u,u/(1+w*u*u)]),('pairwise',.05,5.,lambda u,w:np.c_[np.ones(len(u)),u,*[(u[:,i]+sign*w*u[:,j])**2 for i in range(4) for j in range(i+1,4) for sign in (-1,1)]])]:
        e = []
        for fold in range(4):
            a = np.arange(len(u))%4 != fold
            def resid(w):
                f = fn(u,w)
                return np.mean((f[a]@m._fit(f[a],v[a])-v[a])**2)
            w = minimize_scalar(resid,bounds=(lo,hi),method='bounded').x
            f = fn(u,w)
            e.append(np.mean((f[~a]@m._fit(f[a],v[a])-v[~a])**2)**.5)
        scores.append((label,np.round(e,6).tolist()))
    print(k,scores)
scalar,g,h = m._design(s)
y,t,x = (s['train'][k] for k in ('y','traj','x'))
clock = s['eval_times']; n=len(x)
scalar,g,h = scalar[:n],g[:n],h[:n]
rng=np.random.default_rng(6041)
folds=[('public52',np.arange(52,n)),('radius',np.argsort(np.linalg.norm((x-x.mean(0))/x.std(0),axis=1))[-16:])]
folds += [(f'axis{i}/{j}',np.argsort(j*x[:,i])[-16:]) for i in range(4) for j in (-1,1)]
folds += [(f'random{i}',rng.permutation(n)[:16]) for i in range(10)]
def timed(a,b,mode):
    def design(i,w):
        f=-np.expm1(-w[0]*clock)
        j=np.log1p(w[1]*clock) if mode=='log' else clock/(1+w[1]*clock)
        return np.concatenate((g[i,None,:]*f[None,:,None],h[i,None,:]*j[None,:,None]),axis=2).reshape(-1,g.shape[1]+h.shape[1])
    def fit(w):
        D=design(a,w)
        return m._fit(D,t[a].ravel())
    w=least_squares(lambda w:design(a,w)@fit(w)-t[a].ravel(),[1.8,1.1 if mode=='log' else .3],bounds=([.02,.001],[20,20])).x
    return (design(b,w)@fit(w)).reshape(len(b),len(clock)),w
results=[]
for label,b in folds:
    a=np.setdiff1d(np.arange(n),b)
    base=m._trajectory(g[a],h[a],t[a],g[b],h[b])
    ablated=m._trajectory(g[a],h[a,:-1],t[a],g[b],h[b,:-1])
    lg,w=timed(a,b,'log'); rat,_=timed(a,b,'rational')
    e=np.array([abs(v-t[b]).mean(axis=0) for v in (base,ablated,lg,rat)])
    results.append(e)
    if label=='public52': print(label,'trajectory MAEs (candidate, no off-diagonal, exp+log, exp+rational)',e.mean(1),'exp+log rates',w)
R=np.array(results)
print('worst fold MAE', R.mean(2).max(0), 'worst fold/horizon MAE', R.max(axis=(0,2)))
