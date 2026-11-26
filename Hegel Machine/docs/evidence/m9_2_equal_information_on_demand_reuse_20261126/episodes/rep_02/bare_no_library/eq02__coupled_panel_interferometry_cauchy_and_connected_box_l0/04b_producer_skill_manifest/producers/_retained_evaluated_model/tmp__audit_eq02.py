import pickle
import runpy
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/tmp')
from test_eq02 import eq, check, panel_trial

root = Path('/holdout_task')
studies_path = root / 'public_eq02_studies.pkl'
validator = root / 'validate_public_schema.py'
print('Public studies present:', studies_path.is_file(), '| official validator present:', validator.is_file(), flush=True)
if validator.is_file():
    sys.path.insert(0, str(root))
    runpy.run_path(str(validator), run_name='__main__')

solve = eq.solve_coupled_interferometry_l0

def audit(study):
    panel = study['panel']
    centers = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    cal = np.asarray(study['calibration_indices'])
    optics = study['interferometry']
    phase = np.angle(optics['signal_phasor'] * np.conj(optics['ref_phasor']))
    h = np.column_stack([np.ones(len(strata)), strata == 1, strata == 2, eq._align(panel), phase])
    y = np.asarray(study['y_assay_raw'])
    if cal.dtype == bool:
        cal = np.flatnonzero(cal)
    folds = []
    for c in np.unique(centers[cal]):
        train, held = cal[centers[cal] != c], cal[centers[cal] == c]
        robust = eq._calibrate(h[train], y[train])
        ols = np.linalg.lstsq(h[train], y[train], rcond=None)[0]
        folds.append(tuple(float(np.median(np.abs(y[held] - h[held] @ beta))) for beta in (robust, ols)))
    print('Held-out-center calibration median absolute residual (Cauchy, OLS):', np.round(folds, 5).tolist(), flush=True)
    start = time.perf_counter()
    out = solve(study)
    elapsed = time.perf_counter() - start
    result = check(study['X_cand'], out['deconfounded_target'], study['lam2'], study['lam0'], study['M'], out['certified_selection'])
    print('Clean-process solve seconds:', round(elapsed, 4), '| leaves/gap/minimum PSD eigenvalue:', result, flush=True)
    return out

if studies_path.is_file():
    with studies_path.open('rb') as fh:
        studies = pickle.load(fh)
    if isinstance(studies, dict):
        studies = [studies] if 'panel' in studies else studies.values()
    for study in studies:
        audit(study)
else:
    print('Running the requested center audit on synthetic data; no public measurements are mounted.', flush=True)
    eq.solve_coupled_interferometry_l0 = audit
    panel_trial(107, 19, .83)
