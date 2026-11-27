import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
import upstream_transducer as up
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    ds = pickle.load(f)[0]['dataset']
cal = ds['calibration_load_cells']
x, y = cal['true_channel_loads'], cal['bridge_raw']
a = up._design(x)
fit = up._fit_forward(x, y)
spikes = np.linalg.norm(a @ fit - y, axis=1) > 1
assert np.any(spikes)
blocks = [(k, f) for k in ('train', 'test', 'chamber_verification_trials') if ds.get(k) is not None for f in ('past_channel_raw', 'hold_channel_raw')]
operating = np.concatenate([ds[k][f].reshape(-1, 4) for k, f in blocks])
observed = up._invert(operating, fit)
clean_fit = up._fit_forward(x[~spikes], y[~spikes])
spike_shift = np.max(np.abs(observed - up._invert(operating, clean_fit)))
print('Observed-spike sensitivity, max operating channel shift:', spike_shift)
assert spike_shift < 0.002

# Keep all mixed-load measurements in the fit and hold out each axis sweep in turn.
axis = np.count_nonzero(x, axis=1) == 1
worst = 0.0
for j in range(4):
    test = axis & (x[:, j] != 0)
    fold_fit = up._fit_forward(x[~test], y[~test])
    score = test & ~spikes
    err = up._invert(y[score], fold_fit) - x[score]
    reduced = fold_fit.copy()
    # Refit the quadratic-only null on the same training data without gross spikes.
    keep = ~test & ~spikes
    reduced[:15] = np.linalg.lstsq(a[keep, :15], y[keep], rcond=None)[0]
    reduced[15:] = 0
    quadratic_rmse = np.sqrt(np.mean((a[score] @ reduced - y[score])**2))
    cubic_rmse = np.sqrt(np.mean((a[score] @ fold_fit - y[score])**2))
    peak = np.max(np.abs(err))
    worst = max(worst, peak)
    print(f'axis {j}: n={score.sum()}, quadratic/cubic forward RMSE={quadratic_rmse:.6g}/{cubic_rmse:.6g}, inverse peak={peak:.6g}, total-load peak={np.max(np.abs(err.sum(axis=1))):.6g}')
    assert cubic_rmse < quadratic_rmse
assert worst < 0.002
out = solve_batch(ds)
assert all(np.all(np.isfinite(out[k])) for k in ('estimated_taus', 'hereditary_weights', 'predicted_strain', 'initial_rate_pred'))
print('Isolated composition passed; worst leave-one-axis-out inverse peak:', worst)
