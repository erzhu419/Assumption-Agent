import pickle
import runpy
import numpy as np
from scipy.optimize import least_squares

m = runpy.run_path('/holdout_task/batch_solver.py')
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
tr = s['train']
x, y, obs = (tr[k] for k in ('x', 'y', 'traj'))
t = s['eval_times']
fit, D = m['_fit'], m['_design']
a = m['_calibrate'](s['panel_a'], x)
b = m['_operator'](s['panel_b'], 'v', a, m['_threshold_power'])
c = m['_operator'](s['panel_c'], 'v1', b, m['_odd_saturation'])
c_no_a = m['_operator'](s['panel_c'], 'v1', m['_operator'](s['panel_b'], 'v', x, m['_threshold_power']), m['_odd_saturation'])
c_no_b = m['_operator'](s['panel_c'], 'v1', a, m['_odd_saturation'])

def designs(z, v):
    scalar = D(z, v, np.sum(z*v, axis=1))
    return scalar, np.column_stack((scalar, z[:,1]*v[:,0]))

base = designs(a,c)
no_a, no_b = designs(x,c_no_a), designs(a,c_no_b)
no_cross = (base[0][:,:-1], base[0][:,:-1])

def predict(ds, it, iv):
    gy, gs = ds
    py = gy[iv] @ fit(gy[it], y[it])
    def modes(r): return -np.expm1(-t[:,None]*r)
    def residual(r):
        h = modes(r)
        return (h @ fit(h, obs[it].T) - obs[it].T).ravel()
    h = modes(least_squares(residual, [.42,1.8], bounds=([.05,.95],[.95,5])).x)
    ps = gs[iv] @ fit(gs[it], obs[it]) @ (h @ np.linalg.pinv(h))
    return py, ps

# Test missing pair symmetries and saturated quotients without using outer labels to select terms.
challenges = {}
coords = np.column_stack((a,c))
for i in range(8):
    for j in range(i+1,8):
        for sign in [-1,1]:
            q = coords[:,i] + sign*coords[:,j]
            challenges[f'({i}{sign:+d}*{j})^2'] = q*q
            for rate in [.2,1.]:
                challenges[f'dot/(1+{rate}*({i}{sign:+d}*{j})^2)'] = base[0][:,-1]/(1+rate*q*q)
labels = ['base',*challenges]
variants = [base]+[tuple(np.column_stack((g,v)) for g in base) for v in challenges.values()]
ids = np.arange(len(x))
rng = np.random.default_rng(706)
p = np.argsort(np.linalg.norm(x,axis=1))
splits = [('ordered-52/20',ids[:52],ids[52:]), ('high-norm',p[:52],p[52:])]
for j in range(3):
    p = rng.permutation(ids)
    splits.append((f'random-{j}',p[:52],p[52:]))

def score(py, ps, iv):
    ds = abs(ps-obs[iv]); return np.array([abs(py-y[iv]).mean(), ds.mean(), ds.mean(0).max()])

results = []
for name,it,iv in splits:
    inner = np.array_split(rng.permutation(it),4)
    def inner_metric(g, target):
        return np.mean([np.mean(abs(g[h] @ fit(g[np.setdiff1d(it,h)], target[np.setdiff1d(it,h)]) - target[h])) for h in inner])
    iy = min(range(len(variants)), key=lambda j:inner_metric(variants[j][0],y))
    iz = min(range(len(variants)), key=lambda j:inner_metric(variants[j][1],obs))
    fold = dict(s, train={k:tr[k][it] for k in ('x','y','traj')}, test={'x':x[iv]})
    pred = m['solve_batch'](fold)
    actual = score(pred['y_hat'],pred['state_pred'],iv)
    np.testing.assert_allclose(predict(base,it,iv)[0],pred['y_hat'],atol=1e-9)
    np.testing.assert_allclose(predict(base,it,iv)[1],pred['state_pred'],atol=1e-9)
    challenge = score(*predict((variants[iy][0],variants[iz][1]),it,iv),iv)
    results.append([actual,challenge,*[score(*predict(d,it,iv),iv) for d in [no_a,no_b,no_cross]]])
    print(name, 'base', actual.round(6), 'challenger', challenge.round(6), 'terms', labels[iy], '/', labels[iz])
for label,row in zip(['base','nested challenger','omit A calibration','bypass B','omit interactions'],np.max(results,axis=0)):
    print('worst',label,row.round(6))
assert np.max(np.asarray(results)[:,0]) < .038
print('study fields:', list(s), '; train fields:', list(tr))
print('trajectory two-mode residual RMSE:', np.linalg.norm(np.linalg.svd(obs,compute_uv=False)[2:])/np.sqrt(obs.size))
