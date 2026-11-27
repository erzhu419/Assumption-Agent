import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])
from unittest.mock import patch

def data(i, times=t):
    return p[i], h[i], x[i], dt, times, D[i], True

params, w = m._fit(data(slice(None)), y)
print('fit', np.exp(params), w)
worst = np.zeros((2, 2))
for g in (lab, st):
    for k in np.unique(g):
        v = g == k
        pp, ww = m._fit(data(~v), y[~v])
        abs_err = np.abs(m._basis(pp, *data(v)) @ ww - y[v])
        for active in (False, True):
            e = abs_err[(D[v] > 0) == active]
            stat = np.array([[np.median(e[:, :-1]), np.median(e[:, -1])],
                             [np.quantile(e[:, :-1], .9), np.quantile(e[:, -1], .9)]])
            worst = np.maximum(worst, stat)
print('worst grouped median and p90 (strain, rate):', worst)
assert np.all(worst[0] < [.02, .025]), worst
small = m._basis(params, *data(slice(None), np.array([0., 1e-6]))) @ w
assert np.allclose(small[:, 0], 0) and np.allclose(small[:, 1]/1e-6, small[:, 2], atol=2e-5, rtol=2e-6)
normal = {'lab_id': np.repeat(np.arange(4), 6), 'cmd_stress': np.tile(np.linspace(-2, 2, 6), 4)}
gain = np.array([1.29, .82, 1.17, .76]); bias = np.array([.12, -.15, .14, -.11]); ids = normal['lab_id']
normal['true_stress'] = gain[ids] * normal['cmd_stress'] + bias[ids]
with patch.object(m, '_prior_calibration', side_effect=AssertionError('unnecessary archive access')):
    assert np.allclose(m._calibration(normal), np.column_stack((gain, bias)))
    normal['true_stress'][[0, 9, 15, 22]] += [5, -8, 6, -7]
    assert np.allclose(m._calibration(normal), np.column_stack((gain, bias)))
sparse = m._calibration(d['calibration_load_cells'])
assert np.max(np.abs(sparse - np.column_stack((gain, bias)))) < 2e-4
assert np.isfinite(x).all()
print('Passed analytic-rate, clean and EMI-spiked calibration, sparse calibration, and alignment checks.')
