import json, pickle, sys
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit_forward, _forward, _inverse
from batch_solver import solve_batch

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
with open('/tmp/hegel_continual_archive/archive_apparatus_log.json') as f:
    log = json.load(f)
for entry in log:
    trace = np.asarray(entry['facility_telemetry_trace'])
    print('apparatus', entry['completed_campaign_short_name'], 'trace median', np.median(trace, axis=0).round(5))

terms = {degree: [()] + [t for k in range(1, degree+1) for t in combinations_with_replacement(range(4), k)] for degree in (2, 3)}
def design(a, degree):
    return np.column_stack([a[:, t].prod(axis=1) for t in terms[degree]])

def challenge(a, y, test, channel, degree, rational):
    d, dt = design(a, degree), design(test, degree)
    q, qt = a[:, channel]**2, test[:, channel]**2
    def fit(alpha):
        m = d / (1 + alpha*q[:, None])
        c = np.linalg.lstsq(m, y, rcond=None)[0]
        return np.sum((m@c-y)**2), c
    alpha = 0.
    if rational:
        opt = minimize_scalar(lambda v: fit(v)[0], bounds=(0., 2.), method='bounded', options={'xatol':1e-9})
        if opt.fun < fit(0.)[0]:
            alpha = opt.x
    return (dt@fit(alpha)[1])/(1+alpha*qt)

for item in items:
    study = item['study']; sweeps = study['sensor_calibration_sweeps']; bank = sweeps['single_bank_sweeps']
    x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
    ids, signal, readout = (bank[k] for k in ('bank_id','ref_signal','raw_readout'))
    single = np.eye(4)[ids]*signal[:, None]
    axis = np.count_nonzero(x, axis=1)==1
    emi = np.max(np.abs(y), axis=1)>8
    joint = np.flatnonzero(~axis)
    peak = np.max(abs(x), axis=1)
    edge = joint[np.argsort(peak[joint])[-len(joint)//4:]]
    folds = [('bank'+str(j), axis & (x[:,j]!=0), ids==j) for j in range(4)]
    folds += [('outer joint quartile', np.isin(np.arange(len(x)),edge), np.zeros_like(ids,dtype=bool))]
    config = [(2,False),(3,False),(2,True),(3,True)]
    worst = np.zeros(6)
    for name, test, banktest in folds:
        tr = ~test
        b = {k: np.asarray(v)[~banktest] for k,v in bank.items()}
        c = _fit_forward(x[tr], y[tr], b)
        take = test & ~emi
        # Train alternatives on the same clean training sweeps, with no access to held-out measurements.
        ok = tr & (np.max(abs(_forward(x,c)-y), axis=1)<.005)
        sx = _forward(single,c)[np.arange(len(ids)),ids]
        single_ok = abs(sx-readout)<.005
        bclean = ~banktest & single_ok
        f = _forward(x[take],c)
        obs_error = np.max(abs(_inverse(y[take],c)-x[take]))
        clean_bank_test = banktest & single_ok
        bank_error = np.max(abs(sx[clean_bank_test]-readout[clean_bank_test]), initial=0.)
        residuals = [np.max(abs(f-y[take])), obs_error]
        for degree, rat in config:
            predictions = []
            for j in range(4):
                sel = bclean & (ids==j)
                predictions.append(challenge(np.r_[x[ok],single[sel]], np.r_[y[ok,j],readout[sel]], x[take], j, degree, rat))
            residuals.append(np.max(abs(np.column_stack(predictions)-y[take])))
        worst = np.maximum(worst,residuals)
        print(name, 'rosette/single heldout clean counts', take.sum(), clean_bank_test.sum(), 'errors sparse F/I, full Q/C, rational Q/C', np.array(residuals), 'single', bank_error)
        assert obs_error < 1e-4 and bank_error < 1e-4
    print('worst', worst, 'public diagnostics', item.get('campaign_diagnostics'), 'pulse present', study.get('reference_tracer_pulses') is not None)
    result = solve_batch(study)
    cert = result['kkt_certificate']
    gap = abs(cert['primal_objective']-cert['dual_lower_bound'])
    assert np.all(np.isfinite(result['terminal_field'])) and gap<1e-10
    print('composed KKT gap',gap)
