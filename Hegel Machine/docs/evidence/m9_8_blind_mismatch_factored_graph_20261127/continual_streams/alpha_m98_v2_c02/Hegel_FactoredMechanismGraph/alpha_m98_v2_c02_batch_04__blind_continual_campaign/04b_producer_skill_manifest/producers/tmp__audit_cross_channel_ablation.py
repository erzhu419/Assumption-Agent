import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _invert, _design, _POWERS

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for item in batch:
    cal = item['dataset']['calibration_load_cells']
    x, y = cal['true_channel_loads'], cal['bridge_raw']
    _, clean = _fit(x, y)
    design = _design(x, _POWERS)
    interaction = np.count_nonzero(_POWERS, axis=1) > 1
    load = np.linalg.norm(x, axis=1)
    high = load >= np.quantile(load[clean], .75)
    mixed = np.count_nonzero(x, axis=1) > 1
    for name, hold in [('upper-load mixed', high & mixed), ('lower-load mixed', ~high & mixed)]:
        train = ~hold
        test = hold & clean
        beta, good = _fit(x[train], y[train])
        a, target = design[train][good], y[train][good]
        ablated = np.zeros_like(beta)
        for k in range(y.shape[1]):
            cols = (beta[:, k] != 0) & ~interaction
            ablated[cols, k] = np.linalg.lstsq(a[:, cols], target[:, k], rcond=None)[0]
        rmse = lambda b: np.sqrt(np.mean((design[test] @ b - y[test])**2))
        inverse_error = np.max(abs(_invert(y[test], beta) - x[test]))
        print(f'Cross-channel ablation, {name}, n={test.sum()}: held-out forward RMSE retained={rmse(beta):.6g}, omitted={rmse(ablated):.6g}; retained inverse max abs={inverse_error:.6g}')
        assert inverse_error < .002 and rmse(beta) < rmse(ablated)

sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
