from pathlib import Path
src=Path('/tmp/hegel_continual_archive/alpha_w02_s00_solver.py').read_text()
start=src[src.index('def _checks'):src.index('def _archive_checks')]
start+='''def _archive_checks(root):
    path = root / 'archive_calibration_history.json'
    if path.is_file():
        yield from reversed(json.loads(path.read_text()))


'''
start+=src[src.index('def _with_prior'):src.index('def _design')]
model=src[src.index('def _design'):src.index('def solve_alpha_w02_s00')]
model=model.replace('past, hold = calibrated','past, hold, preload = calibrated').replace('d = hold[:, None] - past @ mix','carry = preload * np.exp(-remaining[0]/tau)\n    d = hold[:, None] - past @ mix - carry').replace('dd = -past @ (a*mix - ea*b*eb)','dd = -past @ (a*mix - ea*b*eb) - carry*remaining[0]/tau')
model=model.replace('np.log(.20)','np.log(.07)').replace('np.log(8.)','np.log(20.)').replace('for b in grid[i+1:]:','for b in grid:\n            if a == b:\n                continue').replace('for i, a in enumerate(grid[:-1]):','for a in grid:').replace("A[valid]@p[2:]", "np.einsum('ij,j->i', A[valid], p[2:])").replace('B@coef-z',"np.einsum('ij,j->i', B, coef)-z")
model=model[:model.index('    order = np.argsort')]+'    return np.exp(p[:2]), p[2:]\n\n\n'
end='''def _stress(split, calib):
    gain, bias = calib[np.asarray(split['lab_id'])].T
    return (np.asarray(split['past_stress_cmd'])*gain[:, None]+bias[:, None],
            np.asarray(split['stress_hold_cmd'])*gain+bias,
            np.asarray(split['initial_memory_states']))


def _crossing(gap, amp, drift, taus, horizon):
    """First upward threshold hit, including hits between the sampled times."""
    if gap <= 0:
        return 0.

    def level(t):
        return drift*t - np.dot(amp, np.expm1(-t/taus)) - gap

    rates = amp/taus

    def speed(t):
        return drift + np.dot(rates, np.exp(-t/taus))

    # The curvature is a sum of two exponentials with at most one zero.
    pieces = [0., horizon]
    a, b = rates/taus
    denominator = 1./taus[0] - 1./taus[1]
    if np.sign(a)*np.sign(b) == -1 and denominator != 0.:
        inflection = (np.log(abs(a))-np.log(abs(b)))/denominator
        if 0. < inflection < horizon:
            pieces.insert(1, inflection)
    knots = [0.]
    for lo, hi in zip(pieces[:-1], pieces[1:]):
        slo, shi = speed(lo), speed(hi)
        if slo*shi < 0.:
            knots.append(brentq(speed, lo, hi))
        if shi == 0. or hi == horizon:
            knots.append(hi)
    for lo, hi in zip(knots[:-1], knots[1:]):
        if level(hi) >= 0.:
            return brentq(level, lo, hi)
    return horizon


def solve_alpha_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    calib = _calibrate(dataset['calibration_load_cells'])
    X = _features(train, test)
    n = len(train['lab_id'])
    train0, test0 = np.asarray(train['strain_0']), np.asarray(test['strain_0'])
    y = np.c_[np.asarray(train['strain_curves_obs'])-train0[:, None], train['instant_rate_obs']]
    taus, coef = _fit(_stress(train, calib), X[:n], dt, time, y)
    stress = _stress(test, calib)
    design = _design(stress, X[n:], dt, time, np.log(taus))
    change = design[:, :-1] @ coef
    pred = test0[:, None]+change
    amp = design[:, -1, :2]*(taus*coef[:2])
    drift = X[n:] @ coef[2:]
    gap = np.asarray(test['critical_strain'])-test0
    return {'estimated_taus': np.sort(taus), 'predicted_strain': pred,
            'secant_compliance': (pred[:, -1]-test0)/np.maximum(np.abs(stress[1]), .5),
            'hazard_crossing_times': np.array([_crossing(g, a, b, taus, time[-1])
                                              for g, a, b in zip(gap, amp, drift)])}
'''
header='''"""Load-cell calibrated, preloaded hereditary creep for polymer coupons."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


'''
Path('/holdout_task/alpha_w02_batch_02_solver.py').write_text(header+start+model+end)
