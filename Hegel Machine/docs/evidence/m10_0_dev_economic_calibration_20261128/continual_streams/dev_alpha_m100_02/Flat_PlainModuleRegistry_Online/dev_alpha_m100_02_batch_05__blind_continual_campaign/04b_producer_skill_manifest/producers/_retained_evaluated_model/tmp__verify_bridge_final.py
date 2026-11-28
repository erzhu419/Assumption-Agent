import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u
import batch_solver

def forward(x, b):
    return (u._design(x, x[..., u._PARTNER]) * b).sum(-1)

for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    ds = item['dataset']; c = ds['calibration_load_cells']
    t, r = c['true_channel_loads'], c['bridge_raw']
    sw = c['single_channel_sweeps']; clean = np.max(abs(r), axis=1) < 5
    ids = np.arange(len(t)); axis = np.count_nonzero(t, axis=1) == 1
    edge = np.max(abs(t), axis=1) >= 1.45
    for name, groups in [('interleaved', ids % 5), ('blocked', ids * 5 // len(t)), ('load-radius', np.argsort(np.argsort(np.max(abs(t), axis=1))) * 5 // len(t))]:
        error = np.full_like(t, np.nan)
        for f in range(5):
            train = groups != f; st = np.arange(len(sw['lab_id'])) % 5 != f
            b = u._fit_bridge(dict(c, true_channel_loads=t[train], bridge_raw=r[train], single_channel_sweeps={k:v[st] for k,v in sw.items()}))
            m = ~train & clean
            error[m] = u._invert(r[m], b) - t[m]
        print(name, 'axis / mixed / edge / overall', *(np.nanmax(abs(error[m])) for m in (axis, ~axis, edge, clean)))
        assert np.nanmax(abs(error)) < 0.001
    b = u._fit_bridge(c)
    calibrated = u.calibrate_upstream(ds)
    extent = max(2., max(np.max(abs(calibrated[s][k])) for s in ('train','test') for k in ('past_channel_raw','hold_channel_raw')))
    _, g, sn, q, cs, ap, am, xy = b.T
    margin = g - 2.5*abs(sn) - abs(q) - 2*abs(cs) - 10*(abs(ap)+abs(am)) - 2*extent*abs(xy)
    print('operating radius / diagonal dominance lower bound', extent, margin.min())
    assert margin.min() > 0
    grid = np.stack(np.meshgrid(*([np.linspace(-extent,extent,15)]*4)),axis=-1).reshape(-1,4)
    np.testing.assert_allclose(u._invert(forward(grid,b),b),grid,atol=1e-8,rtol=0)
    part = {'past_channel_raw': forward(grid[:12].reshape(2,6,4),b), 'hold_channel_raw': forward(grid[:2],b)}
    trials = u.calibrate_upstream(dict(ds,chamber_verification_trials=[part]))['chamber_verification_trials']
    np.testing.assert_allclose(trials[0]['past_channel_raw'].reshape(-1,4),grid[:12],atol=1e-8,rtol=0)
    np.testing.assert_allclose(trials[0]['hold_channel_raw'],grid[:2],atol=1e-8,rtol=0)
    assert u.calibrate_upstream(dict(ds,chamber_verification_trials=None))['chamber_verification_trials'] is None
    cc = calibrated['calibration_load_cells']
    for a,z in [(cc['bridge_raw'],t),(cc['single_channel_sweeps']['cmd_stress'],sw['true_stress'])]:
        assert np.array_equal(a,z) and not np.shares_memory(a,z)
    for s in ('train','test'):
        for k in ('past_channel_raw','hold_channel_raw'):
            np.testing.assert_allclose(forward(calibrated[s][k],b),ds[s][k],atol=1e-9,rtol=0)
    assert all(np.all(np.isfinite(v)) for v in batch_solver.solve_batch(ds).values())
    print('Isolated contract and 50,625 grid inversions passed')
