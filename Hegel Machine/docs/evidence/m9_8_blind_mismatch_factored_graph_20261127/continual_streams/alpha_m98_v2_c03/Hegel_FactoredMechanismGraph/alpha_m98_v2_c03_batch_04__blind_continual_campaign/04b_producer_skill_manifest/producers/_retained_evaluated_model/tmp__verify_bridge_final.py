import pickle, runpy, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _basis, _fit_bridge, _inverse, calibrate_upstream
D = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
C = D['calibration_load_cells']; X = C['true_channel_loads']; Y = C['bridge_raw']; A = _basis(X)
idx = np.arange(len(X)); oof = np.zeros(len(X)); err = np.full_like(X, np.nan)
axis = np.count_nonzero(X, axis=1) <= 1
for te in np.array_split(np.random.default_rng(21).permutation(len(X)), 8):
    tr = np.setdiff1d(idx, te); b = _fit_bridge(X[tr], Y[tr])
    oof[te] = np.linalg.norm(A[te] @ b - Y[te], axis=1)
    good = te[oof[te] < .05]; err[good] = _inverse(Y[good], b) - X[good]
clean = oof < .05
assert np.flatnonzero(~clean).tolist() == [13,14,15]
assert np.max(np.abs(err[clean])) < .002
for label, m in [('axis',axis),('mixed',~axis),('high-load',np.max(np.abs(X),axis=1)>=1.4)]:
    e = err[m & clean]; print(label, 'OOF max abs', np.max(np.abs(e)), 'worst channel RMSE', np.max(np.sqrt(np.mean(e**2,axis=0))))
blocks = [('high-load band',np.max(np.abs(X),axis=1)>=1.4)]
blocks.extend((f'channel {j+1} direction {sign:+d}', ~axis & (sign*X[:,j] > .7)) for j in range(4) for sign in (-1,1))
worst = 0.
for label, te in blocks:
    b = _fit_bridge(X[~te], Y[~te]); good = te & clean
    peak = np.max(np.abs(_inverse(Y[good],b)-X[good])); worst = max(worst,peak)
    print(label,'withheld',good.sum(),'max abs load error',peak)
assert worst < .002
b = _fit_bridge(X,Y); assert np.any(b[9:15]) and np.any(b[15:])
print('Full bridge coefficients maximum change vs archived candidate:')
import importlib.util
spec = importlib.util.spec_from_file_location('old_bridge','/tmp/.hegel_candidate_archive/v01_600e2960/upstream_transducer.py')
from pathlib import Path
archived = list(Path('/tmp/.hegel_candidate_archive/v01_600e2960').rglob('upstream_transducer.py'))
if archived:
    spec = importlib.util.spec_from_file_location('old_bridge',archived[0]); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    print(np.max(np.abs(b - old._fit_bridge(X,Y))))
result = calibrate_upstream(dict(D,chamber_verification_trials=D['test']))
cc = result['calibration_load_cells']; sweeps = cc['single_channel_sweeps']
assert np.array_equal(cc['bridge_raw'], X) and not np.shares_memory(cc['bridge_raw'], X)
assert np.array_equal(sweeps['cmd_stress'], sweeps['true_stress']) and not np.shares_memory(sweeps['cmd_stress'], sweeps['true_stress'])
assert calibrate_upstream(dict(D,chamber_verification_trials=None))['chamber_verification_trials'] is None
for group in ('train','test','chamber_verification_trials'):
    for key in ('past_channel_raw','hold_channel_raw'):
        raw = D['test' if group=='chamber_verification_trials' else group][key]; x = result[group][key]
        assert x.shape == raw.shape and not np.shares_memory(x, raw)
        assert np.all(x.reshape(-1,4) >= X.min(0)) and np.all(x.reshape(-1,4) <= X.max(0))
        reproj = np.max(np.abs(_basis(x.reshape(-1,4)) @ b - raw.reshape(-1,4)))
        print(group,key,'max reprojection',reproj); assert reproj < 1e-9
        if group=='chamber_verification_trials': assert np.array_equal(x,result['test'][key])
sys.argv = ['/holdout_task/validate_public_schema.py','/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0],run_name='__main__')
