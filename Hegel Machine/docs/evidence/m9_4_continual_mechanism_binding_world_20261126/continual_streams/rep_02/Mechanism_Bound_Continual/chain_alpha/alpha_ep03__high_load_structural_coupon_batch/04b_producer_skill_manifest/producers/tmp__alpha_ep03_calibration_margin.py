import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
cal = d['calibration_load_cells']
lab, cmd, truth = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
tr, te, y, complete = s._prepare(d)
sy = d['yield_stress']
max_perturbation = 0.
flips = 0
for c in np.unique(lab):
    ix = np.flatnonzero(lab == c)
    a = np.column_stack((cmd[ix], np.ones(len(ix))))
    b = np.linalg.lstsq(a, truth[ix], rcond=None)[0]
    jack = np.array([np.linalg.lstsq(np.delete(a, j, axis=0), np.delete(truth[ix], j), rcond=None)[0] for j in range(len(ix))])
    error = truth[ix] - np.einsum('ij,ij->i', a, jack)
    changes = 0
    min_gap = np.inf
    shift = 0.
    for panel, design in ((d['train'], tr), (d['test'], te)):
        rows = panel['lab_id'] == c
        for name, stress in (('past_stress_cmd', design.past), ('stress_hold_cmd', design.hold)):
            x = panel[name][rows].ravel()
            p = stress[rows].ravel()
            np.testing.assert_allclose(p, b[0]*x+b[1], rtol=0, atol=1e-12)
            paths = jack[:, :1]*x + jack[:, 1:]
            changes += np.count_nonzero((np.abs(paths)>sy) != (np.abs(p)>sy))
            shift = max(shift, np.max(np.abs(paths-p)))
            min_gap = min(min_gap, np.min(np.abs(np.abs(p)-sy)))
    flips += changes
    max_perturbation = max(max_perturbation, shift)
    print('lab', c, 'gain/bias', b, 'calibration max held-point error', np.max(np.abs(error)), 'max panel stress shift', shift, 'min yield gap', min_gap, 'activation changes', changes)
print('all labs: activation changes', flips, 'largest jackknife stress shift', max_perturbation)
