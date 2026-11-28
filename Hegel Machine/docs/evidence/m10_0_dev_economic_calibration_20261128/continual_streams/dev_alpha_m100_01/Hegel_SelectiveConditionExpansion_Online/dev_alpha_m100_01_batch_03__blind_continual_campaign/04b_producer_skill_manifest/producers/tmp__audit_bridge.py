import pickle, sys
import numpy as np
from scipy.optimize import root
sys.path.insert(0, '/holdout_task')
import upstream_transducer as up

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
axis = np.count_nonzero(x, axis=1) <= 1
amp = np.max(abs(x), axis=1)
reference = up._fit(x, y)
r = np.linalg.norm(up._forward(x, reference)-y, axis=1)
clean = r < .01
print('Calibration fields:', sorted(cal), '| trial metadata:', sorted(ds['train']))
print('Clean residual maximum and spike residual minimum:', max(r[clean]), min(r[~clean]))

def designs(z, model, a):
    base = up._design(z)
    if model == 'difference':
        # Squares of pairwise sums/differences lie in this quadratic span.
        quad = np.stack([z[:,j]*z[:,k] for j in range(4) for k in range(j,4)], axis=1)
        return [np.column_stack((b, quad)) for b in base]
    if model == 'rational':
        for i, b in enumerate(base):
            b[:,5:] /= (1 + a*z[:,i:i+1]**2)
    return base

def fit(idx, model, a=0):
    # The same train-only robust screen is used for every challenger.
    b = up._fit(x[idx], y[idx])
    rr = np.linalg.norm(up._forward(x[idx], b)-y[idx], axis=1)
    idx = idx[rr < max(1e-8, 8*np.median(rr))]
    if model == 'pooled':
        # Test whether offsets and gains vary by channel but normalized
        # forward/back neighbor mixing and nonlinear terms are shared.
        z = x[idx]
        F = np.stack([np.column_stack((z[:,i],z[:,(i+1)%4],z[:,(i-1)%4],z[:,i]*z[:,(i+1)%4],z[:,i]**3)) for i in range(4)])
        Y = y[idx].T
        Fc = F - F.mean(axis=1,keepdims=True)
        Yc = Y-Y.mean(axis=1,keepdims=True)
        gain = b[np.arange(4),1+np.arange(4)]
        for _ in range(40):
            v = np.linalg.lstsq((Fc * gain[:,None,None]).reshape(-1,5), Yc.ravel(), rcond=None)[0]
            v /= v[0]
            q = Fc @ v
            gain = np.sum(q*Yc,axis=1)/np.sum(q*q,axis=1)
        offset = Y.mean(axis=1)-gain*(F.mean(axis=1) @ v)
        ans = np.zeros((4,7)); i = np.arange(4)
        ans[:,0] = offset; ans[i,i+1] = gain
        ans[i,1+(i+1)%4] = gain*v[1]; ans[i,1+(i-1)%4] = gain*v[2]
        ans[:,5:] = gain[:,None]*v[3:]
        return ans
    return [np.linalg.lstsq(d, yy, rcond=None)[0] for d,yy in zip(designs(x[idx],model,a), y[idx].T)]

def predict(z, model, a, b):
    z = np.atleast_2d(z)
    if model in ('baseline','pooled'):
        return up._forward(z, np.asarray(b))
    return np.column_stack([d@bi for d,bi in zip(designs(z,model,a),b)])

rng = np.random.default_rng(310)
fold = np.empty(len(x),dtype=int)
for flag in (False,True):
    idx = np.flatnonzero(axis == flag)
    fold[rng.permutation(idx)] = np.arange(len(idx)) % 5
models = ('baseline','difference','rational','pooled')
errors = {m:np.zeros_like(x) for m in models}
choices = []
for k in range(5):
    tr = np.flatnonzero(fold != k)
    te = np.flatnonzero((fold == k)&clean)
    score = []
    for a in (0., .0001, .001, .01, .05, .2):
        e = []
        for h in range(3):
            j = np.arange(len(tr)) % 3 == h
            u,w = tr[~j],tr[j]
            b = fit(u,'rational',a)
            delta = np.linalg.norm(predict(x[w],'rational',a,b)-y[w],axis=1)
            # Cap spike losses in the fold objective, not the forward fit.
            e.extend(np.minimum(delta,.01)**2)
        score.append(np.mean(e))
    a_selected = (0., .0001, .001, .01, .05, .2)[np.argmin(score)]
    choices.append(a_selected)
    for m in models:
        a = a_selected if m=='rational' else 0.
        b = fit(tr,m,a)
        guess = up._inverse(y[te], up._fit(x[tr],y[tr]))
        for i,g in zip(te,guess):
            ans = root(lambda z: predict(z,m,a,b)[0]-y[i],g)
            assert np.max(abs(ans.fun)) < 1e-8, (m,i,ans.fun)
            errors[m][i] = ans.x-x[i]
print('Nested rational denominator strengths:', choices)
for m,e in errors.items():
    for label,mask in [('all',clean),('axis',clean & axis),('mixed',clean & ~axis),('edge',clean & (amp > 1.55))]:
        z = e[mask]
        print(f'{m:10s} {label:5s} n={len(z):2d} max_channel={np.max(abs(z)):.8g} max_total={np.max(abs(z.sum(axis=1))):.8g} channel_MAE={np.mean(abs(z)):.8g}')
