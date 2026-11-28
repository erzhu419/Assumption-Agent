import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
x, y, z = (s['train'][k] for k in ('x', 'y', 'traj'))
t = s['eval_times']
def fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]
def lin(a):
    return np.column_stack((np.ones(len(a)), a))
pa = s['panel_a']; u, v = pa['u'], pa['v']
bias, gain = [], []
for a, b in zip(u.T, v.T):
    keep = np.arange(len(a))
    design = lin(a)
    for _ in range(3):
        c = fit(design[keep], b[keep])
        keep = np.argsort(abs(b-design@c))[:len(a)*3//4]
    r = abs(b-design@c)
    keep = r < 8*np.median(r)
    c = fit(design[keep], b[keep]); bias.append(c[0]); gain.append(c[1])
bias, gain = np.array(bias), np.array(gain)
pb = s['panel_b']; B = fit(lin(pb['u']), pb['v'])
raw, inv, forward = x, (x-bias)/gain, x*gain+bias
u = lin(inv)@B
models = {'raw_quad':np.c_[lin(x), *[x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]]}
for name, a in [('B', raw), ('invA_B', inv), ('A_B', forward)]:
    b = lin(a)@B
    models[name+'_shared'] = np.c_[lin(b), b[:,0]*b[:,2]+b[:,1]*b[:,3]]
    models[name+'_separate'] = np.c_[lin(b), b[:,0]*b[:,2], b[:,1]*b[:,3]]

def time_model(a, z, at):
    c = fit(a, z)
    def basis(p):
        return -np.expm1(-np.exp(p)[:,None]*t)
    def res(p):
        g = basis(p)
        return (a@(c-fit(g.T,c.T).T@g)).ravel()
    p = least_squares(res, np.log([.4,1.8]), bounds=(-4,3)).x
    g = basis(p)
    w = fit(g.T,c.T).T
    return at@w@g, np.exp(p), w
rng = np.random.default_rng(190)
order = rng.permutation(len(x)); allrows = np.arange(len(x))
splits = [('fixed',allrows[52:]), *[(f'fold{i}',v) for i,v in enumerate(np.array_split(order,4))],('outer input',np.argsort(np.linalg.norm(x,axis=1))[-18:])]
for name, va in splits:
    tr = np.setdiff1d(allrows, va)
    print(name, {key:round(np.mean(abs(y[va]-m[va]@fit(m[tr],y[tr]))),6) for key,m in models.items()})
    a = lin(u); p, rates, w = time_model(a[tr],z[tr],a[va]); e = abs(p-z[va]); print('state mean/worst horizon',e.mean(),e.mean(0).max(),'rates',rates)
    if name=='fixed':
        print('calibration',bias,gain,'test extent',s['test']['x'].min(0),s['test']['x'].max(0))
        print('label coeffs and residual std',[(k,fit(m[tr],y[tr]).round(5).tolist(),round(np.std(y[tr]-m[tr]@fit(m[tr],y[tr])),6)) for k,m in models.items() if k.endswith('separate')]); print('temporal amplitudes, mixed coordinates',w.round(5))
