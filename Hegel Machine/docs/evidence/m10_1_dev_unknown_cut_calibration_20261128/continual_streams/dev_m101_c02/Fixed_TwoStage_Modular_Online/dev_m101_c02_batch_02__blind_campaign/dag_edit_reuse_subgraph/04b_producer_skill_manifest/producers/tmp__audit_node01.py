import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

def load(path):
    scope = {}
    exec(compile(Path(path).read_text(), path, 'exec'), scope)
    return scope

frozen = load('_frozen_nodes/node_00.py')['run_node']
node = load('node_01.py')

def candidate(z, y, ze, kind):
    B, E = node['_scalar_basis'](z), node['_scalar_basis'](ze)
    if kind == 'sum':
        B = np.column_stack((B[:, :5], B[:, 5] + B[:, 6]))
        E = np.column_stack((E[:, :5], E[:, 5] + E[:, 6]))
    Q, _ = np.linalg.qr(B, mode='reduced')
    def orth(x):
        return x - Q @ (Q.T @ x)
    r = orth(y)
    d = np.linalg.lstsq(B, y, rcond=None)[0][1:5]
    d /= np.linalg.norm(d)
    def extra(x,p):
        q = x @ p[:4] + p[4]
        den = 1 + q*q
        cols = [q/den]
        if kind == 'even_projection':
            cols.append(q*q/den)
        if kind == 'gated_pairs':
            cols.extend([x[:,0]*x[:,2]/den, x[:,1]*x[:,3]/den])
        if kind == 'cross_pairs':
            cols.extend([(x[:,0]+x[:,1])*(x[:,2]+x[:,3]), (x[:,0]-x[:,1])*(x[:,2]-x[:,3])])
        return np.column_stack(cols)
    def residual(p):
        H = orth(extra(z,p))
        return H @ np.linalg.lstsq(H,r,rcond=None)[0] - r
    p = min((least_squares(residual, np.r_[a*d,b], bounds=(-5.,5.), max_nfev=300)
             for a in (.6,1.4,2.5) for b in (0.,.25,-.25)), key=lambda result:result.cost).x
    w = np.linalg.lstsq(np.column_stack((B,extra(z,p))), y, rcond=None)[0]
    return np.column_stack((E, extra(ze,p))) @ w

with open('public_batch.pkl','rb') as f:
    batch = pickle.load(f)
print('studies', len(batch))
for k, entry in enumerate(batch):
    s = entry['study']
    print('study', k, 'fields', sorted(s), 'telemetry', s.get('telemetry_vec'))
    tr=s['train']; z=frozen({'__input__':s})['z_tr']; y=tr['y']; idx=np.arange(len(y))
    direction=np.linalg.lstsq(node['_scalar_basis'](z),y,rcond=None)[0][1:5]
    order=np.argsort(z@direction)
    folds={'official':idx[52:72], **{f'block_{i}':j for i,j in enumerate(np.array_split(idx,6))},
           'radius':np.argsort(np.linalg.norm(z,axis=1))[-12:], 'low_projection':order[:12], 'high_projection':order[-12:]}
    names=('current','sum','even_projection','gated_pairs','cross_pairs')
    print('slice',*names,'trajectory')
    for name, v in folds.items():
        if not len(v):
            continue
        f=np.setdiff1d(idx,v)
        pred=[node['_predict_scalar'](z[f],y[f],z[v]), *(candidate(z[f],y[f],z[v],kind) for kind in names[1:])]
        A=node['_affine']
        tp=A(z[v])@np.linalg.lstsq(A(z[f]),tr['traj'][f],rcond=None)[0]
        print(name,*(f'{np.mean(abs(p-y[v])):.6f}' for p in pred),f'{np.mean(abs(tp-tr["traj"][v])):.6f}')
