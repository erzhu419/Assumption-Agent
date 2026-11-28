import pickle
import sys
from itertools import combinations_with_replacement
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as m

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
sw = c['single_channel_sweeps']
print('Dataset keys:', list(d), 'calibration keys:', list(c), 'sweep counts:', np.unique(sw['lab_id'], return_counts=True))
j = np.arange(4); nxt = (j+1)%4; prev = (j-1)%4

def features(x, mode):
    n = len(x)
    z = x[:, nxt]
    base = np.stack((np.ones_like(x), x, z, x[:, prev], x**3, x*z), axis=1)
    if mode == 'dense':
        terms = [t for k in range(4) for t in combinations_with_replacement(range(4), k)]
        return np.broadcast_to(np.column_stack([np.prod(x[:,t], axis=1) for t in terms])[:,:,None], (n,len(terms),4))
    if mode == 'rational':
        # Complement the polynomial with saturation corrections in the self and relative-load coordinates.
        q = x-z
        return np.concatenate((base, np.stack((x**5/(1+0.5*x*x), q**3/(1+0.5*q*q)), axis=1)), axis=1)
    return base

def augmented_fit(a, b):
    w = np.ones(len(b))
    for _ in range(40):
        coef = np.column_stack([np.linalg.lstsq(a[:,:,k]*w[:,None], b[:,k]*w, rcond=None)[0] for k in j])
        e = np.linalg.norm(np.sum(a*coef, axis=1)-b, axis=1)
        w = 1/np.hypot(1, e/max(2.5*np.median(e),1e-9))
    return coef

# Blocked extrapolation slices supplement the previous randomized eight-fold test.
blocks = {'single-axis':np.count_nonzero(x,axis=1)==1,
          'outer-mixed':(np.count_nonzero(x,axis=1)>1)&(np.abs(x).max(axis=1)>1.4)}
blocks.update({f'near-cancel-{k}': (np.abs(x[:,k]+x[:,nxt[k]])<0.4)&(np.abs(x[:,k])+np.abs(x[:,nxt[k]])>0.7) for k in j})
for name, mask in blocks.items():
    tr, te = ~mask, mask
    if not te.any():
        continue
    coef = m._fit_bridge(x[tr],y[tr]); base = np.linalg.norm(m._forward(x[te],coef)-y[te],axis=1)
    clean = base<1
    inv = np.max(np.abs(m._inverse(y[te][clean],coef)-x[te][clean]))
    errors = {'cyclic':float(np.max(base[clean]))}
    for mode in ('dense','rational'):
        a,b = features(x[tr],mode),features(x[te][clean],mode)
        e = np.linalg.norm(np.sum(b*augmented_fit(a,y[tr]), axis=1)-y[te][clean],axis=1)
        errors[mode] = float(np.max(e))
    print(name, 'clean n', int(clean.sum()), 'spikes', int((~clean).sum()), 'forward max',errors,'cyclic inverse max',inv)

coef = m._fit_bridge(x,y)
stress = sw['true_stress']; ch = sw['lab_id']; i = np.arange(len(stress)); loads = np.zeros((len(stress),4)); loads[i,ch] = stress
e = np.abs(m._forward(loads,coef)[i,ch]-sw['cmd_stress'])
for k in j:
    use = ch==k; clean = use&(e<1)
    print('independent sweep channel',k,'clean max error',float(e[clean].max()),'spikes',int(np.sum(use&(e>=1))))
