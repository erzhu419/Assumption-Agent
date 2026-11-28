import sys, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from batch_solver import _nodes
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
p = s['train']; a, b, c = _nodes(s, p['x']); y, tr = p['y'], p['traj']; t = s['eval_times']; n = len(y)
z = a*c; base = np.c_[np.ones(n), a, c]; off = a[:,1]*c[:,0]
S = [np.c_[base,z], np.c_[base,z.sum(1)]]
T = [np.c_[base,(a[:,:,None]*c[:,None,:]).reshape(n,-1)], np.c_[base,z,off], np.c_[base,z.sum(1),off]]
def fit(g,i,j,v):
    return g[j] @ np.linalg.lstsq(g[i],v[i],rcond=None)[0]
def basis(r):
    return -np.expm1(-t[:,None]*np.asarray(r))
def prof(v):
    def res(r):
        u=basis(r); return (u @ np.linalg.lstsq(u,v.T,rcond=None)[0]-v.T).ravel()
    return least_squares(res,[1.8,.4],bounds=([1.,.1],[3.,.9])).x
idx=np.arange(n); rng=np.random.default_rng(425)
splits=[('52/20',idx[:52],idx[52:]),('56/16',idx[:56],idx[56:])]
for key,q in [('radius',np.linalg.norm(a,axis=1)),('coordinate',abs(a).max(1))]:
    o=np.argsort(q); splits.append((key,o[:56],o[56:]))
o=np.argsort(np.sum(abs(a)>.35,axis=1)); splits.append(('low-active',o[16:],o[:16]))
splits += [('random',o[:52],o[52:]) for o in (rng.permutation(idx) for _ in range(30))]
results={}; rates=[]
for label,i,j in splits:
    r=prof(tr[i]); rates.append(r)
    P=[np.linalg.svd(tr[i],full_matrices=False)[2][:2], np.linalg.qr(basis(r))[0].T, np.linalg.qr(basis([1.8,.4]))[0].T]
    for k,g in enumerate(S):
        e=abs(fit(g,i,j,y)-y[j]); results.setdefault(('scalar',k),[]).append([e.mean()])
    for k,g in enumerate(T):
        pred=fit(g,i,j,tr)
        for d,u in enumerate(P):
            e=abs(pred@u.T@u-tr[j]); results.setdefault(('traj',k,d),[]).append([e.mean(), e.mean(0).max()])
print('projection rates overall',prof(tr),'fold range',np.min(rates,0),np.max(rates,0))
print('rows: 52/20, 56/16, high radius, high coordinate, low active, random worst; trajectory columns: overall, worst timestamp')
for k,v in results.items():
    v=np.array(v); print(k,np.round(np.vstack([v[:5],v[5:].max(0)]),6).tolist())
