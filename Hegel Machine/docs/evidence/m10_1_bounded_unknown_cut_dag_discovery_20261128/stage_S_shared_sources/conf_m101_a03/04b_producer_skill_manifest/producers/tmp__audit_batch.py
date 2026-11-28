import importlib.util
import pickle
from itertools import combinations_with_replacement

import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_batch.pkl', 'rb') as fh:
    s = pickle.load(fh)[0]['study']
t = s['train']; x, y, q = (t[k] for k in ('x','y','traj'))
rows = np.arange(len(x))
folds = [('52/20',rows[52:])]
folds += [(f'outer_x{k}',np.argsort(abs(x[:,k]-np.median(x[:,k])))[-18:]) for k in range(4)]
folds += [('outer_norm',np.argsort(np.linalg.norm((x-x.mean(0))/x.std(0),axis=1))[-18:])]
folds += [(f'random_{k}',j) for k,j in enumerate(np.array_split(np.random.default_rng(42).permutation(rows),4))]
b, g = m._calibration(s['panel_a'])
z = m._affine((x-b)/g) @ m._fit(m._affine(s['panel_b']['u']),s['panel_b']['v'])
a = m._affine(z)
pairs = np.array(list(combinations_with_replacement(range(4),2)))
p = z[:,pairs[:,0]]*z[:,pairs[:,1]]
full = np.column_stack((a,p))
# Quotient directions include each channel, the sum/difference of each pair,
# and the pooled interaction itself. Tune within each outer training set.
directions = [z[:,k] for k in range(4)]
directions += [z[:,i]+sgn*z[:,j] for i in range(4) for j in range(i+1,4) for sgn in (-1,1)]

def design(i):
    keep = np.argsort(abs(m._fit(full[i],y[i])[5:]))[-2:]
    return np.column_stack((a,p[:,keep].sum(1))), pairs[keep]

def rational(w, idx, direction, strength):
    v = (p@((np.arange(p.shape[1])[:,None]==np.array([2,6])).sum(1))) if direction == len(directions) else directions[direction]
    scale = np.mean(v[idx]**2)
    return w/(1+strength*v[:,None]**2/scale)

def mae(pred, j):
    return abs(pred-y[j]).mean()

stats=[]
print('fold | selected pairs | affine / full quad / solver / tuned quotient scalar MAE | solver worst horizon | quotient strength')
for label,j in folds:
    i = np.setdiff1d(rows,j)
    subset = {'x':x[i],'y':y[i],'traj':q[i],'summary':np.array([y[i].mean(),y[i].std(),q[i,-1].mean(),q[i,-1].std()])}
    out = m.solve_batch(dict(s, train=subset, test={'x':x[j]}))
    w, terms = design(i)
    inner = [(np.setdiff1d(i,v),v) for v in np.array_split(np.random.default_rng(83).permutation(i),3)]
    inner = [(u,v,design(u)[0]) for u,v in inner]
    def inner_loss(direction,strength):
        res = []
        for u,v,base in inner:
            r = rational(base,u,direction,strength)
            res.extend(abs(r[v]@m._fit(r[u],y[u])-y[v]))
        return np.mean(res)
    options = [(0,0.)]+[(d,h) for d in range(len(directions)+1) for h in (0.0003,0.001,0.003,0.01,0.03,0.1)]
    direction,strength = min(options,key=lambda dh:inner_loss(*dh))
    r = rational(w,i,direction,strength)
    metric = [mae(a[j]@m._fit(a[i],y[i]),j),mae(full[j]@m._fit(full[i],y[i]),j),mae(out['y_hat'],j),mae(r[j]@m._fit(r[i],y[i]),j),abs(out['state_pred']-q[j]).mean(0).max()]
    stats.append(metric)
    print(label,tuple(map(tuple,terms)),*(round(v,6) for v in metric),strength)
print('worst:',np.max(stats,axis=0))
print('available training fields:',sorted(t),'telemetry shape:',s['telemetry_vec'].shape)
assert np.max(np.array(stats)[:,[2,4]]) < .038
