import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
b = m._calibrate(d['calibration_load_cells'])
tr, p = [d[k] for k in ('train', 'chamber_verification_trials')]
dt, t = d['past_dt'], d['eval_times']
train = m._prepare(tr, b)
r = tr['ref_coupons']
ref = {g: (np.median(r[tr['lab_id']==g].mean(0)), np.median(r[tr['lab_id']==g].std(0))) for g in np.unique(tr['lab_id'])}
def take(a, j):
    return {k:v[j] for k,v in a.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(j)}
def stats(e):
    return [np.median(abs(e[:,0])), np.median(abs(e[:,1:])), np.median(abs(e[:,-1]))]
errors = {'split_local': [], 'benchtop_panel': [], 'no_pilot': []}
bench = m._estimate(train, None, dt, t)
for lab in np.unique(p['lab_id']):
    j = p['lab_id'] == lab
    for name in errors:
        panel = ref if name == 'benchtop_panel' else None
        held = m._prepare(take(p,j), b, reference=panel)
        fit = bench if name == 'no_pilot' else m._estimate(train, m._prepare(take(p,~j),b,reference=panel),dt,t)
        err = m._design(held, fit[0], dt,t) @ fit[1] - held[4]
        errors[name].append(stats(err))
for name, rows in errors.items():
    print(name, 'worst held-out chamber lab median [rate, strain, terminal]:', np.max(rows,axis=0))
assert np.max(errors['split_local'],axis=0)[1] < np.max(errors['benchtop_panel'],axis=0)[1]

# Apply a known cross-talk and offset change to all raw samples and to the
# calibration sweeps. Dynamic calibration should preserve physical outputs.
A = np.array([[1.16,.12,-.06,.04],[-.09,.81,.13,.08],[.04,-.07,1.24,.06],[.11,.05,-.1,.91]])
c = np.array([.19,-.28,.08,.23])
changed = dict(d)
cal = d['calibration_load_cells']
changed['calibration_load_cells'] = {**cal, 'bridge_raw': cal['bridge_raw'] @ A + c}
for key in ('train','test','chamber_verification_trials'):
    q = d[key]
    changed[key] = {**q, **{k:q[k] @ A + c for k in ('past_channel_raw','hold_channel_raw')}}
for pilot in (True,False):
    a,z = d,changed
    if not pilot:
        a,z = ({**q,'chamber_verification_trials':None} for q in (a,z))
    x,y = m.solve_batch(a),m.solve_batch(z)
    diffs = {k:float(np.max(abs(x[k]-y[k]))) for k in x}
    print('affine bridge invariance', 'pilot' if pilot else 'blind', diffs)
    assert all(np.allclose(x[k],y[k],atol=2e-5,rtol=2e-5) for k in x)
