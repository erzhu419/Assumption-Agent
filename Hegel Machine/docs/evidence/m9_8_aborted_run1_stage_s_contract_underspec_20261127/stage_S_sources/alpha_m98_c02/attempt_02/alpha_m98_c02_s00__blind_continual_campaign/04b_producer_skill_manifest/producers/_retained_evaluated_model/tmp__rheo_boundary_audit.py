import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
t, dt = d['eval_times'], d['past_dt']
b = m._load_bridge(d['calibration_load_cells'])
base = m._design

def equilibrium(q, t, dt, tau):
    a, r = base(q, t, dt, tau)
    extra = q[1][:, 0, None] * np.exp(-dt.sum() / tau)
    a[:, :, :2] -= extra[:, None, :] * -np.expm1(-t[None, :, None] / tau)
    r[:, :2] -= extra / tau
    return a, r

for cohort in ('chamber_verification_trials', 'train'):
    g = d[cohort]
    q = m._prepare(g, b)
    labs = g['lab_id']
    y = g['strain_curves_obs'] - g['strain_0'][:, None]
    r = g['instant_rate_obs']
    for name, design in [('delivered_zero_state', base), ('equilibrated_first_load', equilibrium)]:
        m._design = design
        e, er = np.empty_like(y), np.empty_like(r)
        for lab in np.unique(labs):
            sel = labs == lab
            tau, w = m._fit(g, q, t, dt, pilot=cohort != 'train', mask=~sel)
            a, v = design(q, t, dt, tau)
            e[sel], er[sel] = a[sel] @ w - y[sel], v[sel] @ w - r[sel]
        def metric(sel):
            return np.round([np.quantile(abs(e[sel]), .9), np.quantile(abs(er[sel]), .9), np.quantile(abs(e[sel, -3:]), .9)], 6)
        print(cohort, name, 'complete', metric(q[3]), 'missing', metric(~q[3]), 'worst_lab', np.max([metric(labs == l) for l in np.unique(labs)], axis=0))
