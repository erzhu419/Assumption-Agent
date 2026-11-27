import importlib.util
import pickle
import numpy as np
p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p+'batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(p+'public_batch.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
xt, xe = s._features(tr, te)
cal = s._calibration(d['calibration_load_cells'])
a, b = s._inputs(tr, xt, cal), s._inputs(te, xe, cal)
tau, th, scale = s._material(tr, a, dt, t)
y, r = s._predict(b, dt, t, tau, th)
base = s.solve_batch(d)
rng = np.random.default_rng(19)
for repeat in range(3):
    idx, held = np.split(rng.permutation(len(xe)), 2)
    pi = {k:v[idx].copy() for k,v in te.items()}
    pi['coupon_features_raw'] = xe[idx].copy()
    pi['ref_coupons'] = np.empty((len(idx), 8))
    for lab in np.unique(pi['lab_id']):
        mask = pi['lab_id'] == lab
        z = np.arange(mask.sum(), dtype=float)
        z -= z.mean()
        pi['ref_coupons'][mask] = (z/z.std())[:,None] if z.std() else 0.
    xp, = s._features(pi, balanced=False)
    np.testing.assert_allclose(xp, xe[idx], atol=1e-12)
    pin = s._inputs(pi, xp, cal)
    for mem, drift, c in [(None,None,[]), ('initial',None,[-.8,.4]), (None,1.5,[.8,.15,.2,-.04]), ('order','linear',[.7,.1,-.3])]:
        delta = s._chamber_design(b,dt,t,tau,th,mem,drift) @ c
        pi['strain_curves_obs'] = pi['strain_0'][:,None]+y[idx]+delta[idx,:-1]
        pi['instant_rate_obs'] = r[idx]+delta[idx,-1]
        choice = s._pilot_choice(pin,pi,dt,t,tau,th,scale)
        out = s.solve_batch(dict(d,chamber_verification_trials=pi))
        ec = out['predicted_strain'][held]-te['strain_0'][held,None]-y[held]-delta[held,:-1]
        er = out['initial_rate_pred'][held]-r[held]-delta[held,-1]
        rm = lambda z: float(np.sqrt(np.mean(z*z)))
        print(repeat, (mem, drift), 'fit', choice[:2], 'strain', round(rm(delta[held,:-1]),6), '->', round(rm(ec),6), 'rate', round(rm(delta[held,-1]),6), '->', round(rm(er),6))
        for k in ('estimated_taus','hereditary_weights'):
            np.testing.assert_array_equal(base[k], out[k])
