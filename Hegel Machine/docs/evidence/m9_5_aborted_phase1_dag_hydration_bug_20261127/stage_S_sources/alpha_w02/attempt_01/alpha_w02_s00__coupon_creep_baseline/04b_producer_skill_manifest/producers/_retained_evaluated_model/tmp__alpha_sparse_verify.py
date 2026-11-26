import copy, importlib.util, json, pickle, tempfile
from pathlib import Path
import numpy as np
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_w02_s00_solver.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
with (root/'public_alpha_w02_s00.pkl').open('rb') as f: case = pickle.load(f)[0]
d, g = case['dataset'], case['gt']
o0 = s.solve_alpha_w02_s00(d)
truth = np.array([[1.08,.06],[.92,-.05],[1.11,-.08],[.89,.09]])
def batch(labs, count, glitch=False):
    b = copy.deepcopy(d)
    c = b['calibration_load_cells']; keep = np.ones(len(c['lab_id']), bool)
    target = s._calibration(d)
    for l in labs:
        a,e = truth[l]; target[l] = a,e
        for split in ('train','test'):
            q = b[split]; j = q['lab_id']==l
            for k in ('past_stress_cmd','stress_hold_cmd'): q[k][j] = (q[k][j]-e)/a
        j = np.flatnonzero(c['lab_id']==l); j = j[np.argsort(c['cmd_stress'][j])]
        chosen = j[np.round(np.linspace(0,len(j)-1,count)).astype(int)]
        keep[j] = False; keep[chosen] = True
        c['cmd_stress'][j] = (c['cmd_stress'][j]-e)/a
        if glitch: c['true_stress'][chosen[count//2]] += .15
    b['calibration_load_cells'] = {k:v[keep] for k,v in c.items()}
    return b,target

def check(label, b, target, tol):
    x,_,m = s._align(b); cal = s._sparse_calibration(b,s._calibration(b),x,m)
    stress = float(np.max(np.abs(cal-target) @ [2.5,1.]))
    out = s.solve_alpha_w02_s00(b)
    changes = [np.max(np.abs(out[k]-o0[k])) for k in ('predicted_strain','initial_rate_pred')]
    print(label, 'stress error', round(stress,6), 'max curve/rate changes', np.round(changes,6))
    assert stress < tol and max(changes) < .03
    assert .2 <= out['estimated_taus'][0] < out['estimated_taus'][1] <= 8
    np.testing.assert_array_equal(out['viscoplastic_weights'], 0)

check('both wings, two clean checks', *batch(range(4),2), .001)
for wing in ((0,1),(2,3)):
    for n, glitch in ((1,False),(2,False),(4,True)):
        check((wing,n,glitch), *batch(wing,n,glitch), .02)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    a = batch(range(4),10)[0]['calibration_load_cells']
    (path/'calibration.json').write_text(json.dumps({'calibration_load_cells': {k:v.tolist() for k,v in a.items()}}))
    s.Path = lambda _: path
    for glitch in (False,True):
        check(('both wings with archive, one check',glitch), *batch(range(4),1,glitch), .001)
print('Public tau absolute errors:', np.round(np.abs(o0['estimated_taus']-g['taus']),6))
print('Public maximum hereditary coefficient error:', round(float(np.max(np.abs(o0['hereditary_weights']-g['theta']))),6))
