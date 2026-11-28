import pickle, runpy
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations

root = '/holdout_task/'
node = runpy.run_path(root + 'node_00.py')
with open(root + 'public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
s = batch[0]['study']
print('Public study count and schema:', len(batch), {k: sorted(v) if isinstance(v, dict) else type(v).__name__ for k,v in s.items()})
u, v = (s['panel_b'][k] for k in ('u', 'v'))
pairs = list(combinations(range(u.shape[1]), 2))
fit = node['_fit']
aff = node['_affine']

def search(loss, grid):
    i = np.argmin([loss(w) for w in grid])
    return minimize_scalar(loss, bounds=(grid[max(0,i-1)], grid[min(len(grid)-1,i+1)]), method='bounded').x

def train(x, y, mode):
    if mode == 'sine':
        w, b = node['_mixer']({'u': x, 'v': y})
        return lambda t: node['_mixer_basis'](t,w) @ b
    def make(t, w, form=None):
        base = node['_mixer_basis'](t,w)
        if mode == 'square': return np.c_[base,t*t]
        if mode == 'cosine': return np.c_[base,np.cos(w*t)]
        if mode == 'pair':
            i,j,sign = form
            return np.c_[base,np.sin(w*(t[:,i]+sign*t[:,j]))]
        if mode == 'rational':
            a, q = form
            return base / (1+a*q(t)**2)[:,None]
        return aff(t)
    if mode == 'affine':
        b = fit(aff(x),y)
        return lambda t: aff(t) @ b
    w0, _ = node['_mixer']({'u': x, 'v': y})
    forms = [None]
    if mode == 'pair': forms = [(i,j,sign) for i,j in pairs for sign in (-1,1)]
    if mode == 'rational':
        coords = [lambda t,i=i: t[:,i] for i in range(x.shape[1])]
        diffs = [lambda t,i=i,j=j: (t[:,i]-t[:,j])/np.sqrt(2) for i,j in pairs]
        forms = [(a,q) for a in (.01,.05,.2) for q in coords+diffs+[lambda t: np.linalg.norm(t,axis=1)/2]]
    def loss(w, form):
        A = make(x,w,form)
        return np.sum((y-A@fit(A,y))**2)
    form = min(forms,key=lambda f: loss(w0,f))
    w = search(lambda w: loss(w, form), np.linspace(.5,5.,19))
    b = fit(make(x,w,form),y)
    return lambda t: make(t,w,form) @ b

for mode in ('affine','sine','square','cosine','pair','rational'):
    errors = []
    for i in range(len(u)):
        keep = np.arange(len(u)) != i
        predict = train(u[keep],v[keep],mode)
        errors.append(abs(predict(u[i:i+1])[0]-v[i]))
    e = np.array(errors)
    print(f'{mode:8s} panel B nested LOOCV: MAE={e.mean():.6f}, worst output MAE={e.mean(axis=0).max():.6f}, worst row MAE={e.mean(axis=1).max():.6f}')
