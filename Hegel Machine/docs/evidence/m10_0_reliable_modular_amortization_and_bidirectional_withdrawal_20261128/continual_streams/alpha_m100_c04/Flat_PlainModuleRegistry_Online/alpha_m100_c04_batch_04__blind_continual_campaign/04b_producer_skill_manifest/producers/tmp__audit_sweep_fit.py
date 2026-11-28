import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as m
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']; x, y = c['true_channel_loads'], c['bridge_raw']
s = c['single_channel_sweeps']; u, v, lab = (s[k] for k in ('true_stress', 'cmd_stress', 'lab_id'))
j = np.arange(4); nxt = (j+1)%4; prev = (j-1)%4

def joint_fit(x, y, u, v, lab):
    a = np.stack((np.ones_like(x), x, x[:,nxt], x[:,prev], x**3, x*x[:,nxt]), axis=1)
    b = np.column_stack((np.ones_like(u), u, np.zeros_like(u), np.zeros_like(u), u**3, np.zeros_like(u)))
    coeff = m._fit_bridge(x,y)
    for _ in range(30):
        r = np.linalg.norm(m._forward(x,coeff)-y,axis=1)
        e = abs(np.sum(b*coeff[:,lab].T,axis=1)-v)
        scale = max(np.median(r)*2.5,1e-9)
        w = 1/np.hypot(1,r/scale); ws = 1/np.hypot(1,e/scale)
        coeff = np.column_stack([np.linalg.lstsq(np.vstack((a[:,:,k]*w[:,None],b[lab==k]*ws[lab==k,None])),np.r_[y[:,k]*w,v[lab==k]*ws[lab==k]],rcond=None)[0] for k in j])
    return coeff

spikes = np.isin(np.arange(len(x)), [13,14,15])
mixed = np.count_nonzero(x,axis=1)>1
folds = [('random-'+str(i),mask) for i,ids in enumerate(np.array_split(np.random.default_rng(72).permutation(len(x)),8)) for mask in [np.isin(np.arange(len(x)),ids)]]
folds += [('single-axis',~mixed),('outer-mixed',mixed&(abs(x).max(1)>1.4))]
folds += [(f'near-cancel-{k}',(abs(x[:,k]+x[:,nxt[k]])<.4)&(abs(x[:,k])+abs(x[:,nxt[k]])>.7)) for k in j]
maxes = np.zeros((2,2))
for name, te in folds:
    tr=~te; keep=te&~spikes
    # Exclude sweep points at held-out single-axis calibration coordinates.
    sx = np.zeros((len(u),4)); sx[np.arange(len(u)),lab]=u
    use = ~np.isclose(sx[:,None],x[te][None]).all(axis=2).any(axis=1)
    models = (m._fit_bridge(x[tr],y[tr]),joint_fit(x[tr],y[tr],u[use],v[use],lab[use]))
    err = np.array([np.max(abs(m._inverse(y[keep],p)-x[keep]),axis=0) for p in models])
    index = int(not name.startswith('random-'))
    maxes[index] = np.maximum(maxes[index],err.max(1))
    if index: print(name,'baseline vs sweep-augmented per-channel inverse maxima',err.tolist())
print('Worst clean inverse error, columns baseline and sweep-augmented:',dict(zip(('randomized','blocked'),maxes.tolist())))
