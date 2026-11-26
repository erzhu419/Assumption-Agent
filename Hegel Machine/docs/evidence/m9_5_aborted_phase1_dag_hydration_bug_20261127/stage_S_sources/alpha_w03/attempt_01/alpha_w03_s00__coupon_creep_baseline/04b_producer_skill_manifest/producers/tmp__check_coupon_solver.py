from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('coupon', root/'alpha_w03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_w03_s00.pkl').open('rb') as f:
    p = pickle.load(f)[0]
d, gt = p['dataset'], p['gt']
keys = ('lab_id', 'cmd_stress', 'true_stress')
base = d['calibration_load_cells']

def validate(name, dd):
    out = m.solve_alpha_w03_s00(dd)
    tau = out['estimated_taus']
    assert .2 <= tau[0] < tau[1] <= 8 and np.all(out['viscoplastic_weights'] == 0)
    assert all(np.isfinite(v).all() for v in out.values())
    c = out['predicted_strain'] - gt['te_clean']
    r = out['initial_rate_pred'] - gt['te_true_rate0']
    te = dd['test']
    missing = np.isnan(te['coupon_features_raw']).any(axis=1)
    groups = [te['lab_id']==i for i in range(4)] + [te['stratum']==j for j in np.unique(te['stratum'])] + [missing, ~missing]
    rmse = lambda a: [np.sqrt(np.mean(a[g]**2)) for g in groups if g.any()]
    cs, rs = rmse(c), rmse(r)
    print(name, 'worst curve/rate', np.round([max(cs),max(rs)], 5), 'max relative tau / abs weight error', np.round([max(abs(tau/gt['taus']-1)),max(abs(out['hereditary_weights']-gt['theta']))],5), 'per-lab curve', np.round(cs[:4],5), flush=True)

validate('public', d)
with TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp)
    for wing in ((0,1),(2,3)):
        shifted = deepcopy(d)
        truth = {wing[0]:(.86,.10), wing[1]:(1.14,-.10)}
        for split in ('train','test'):
            s = shifted[split]
            for lab,(a,b) in truth.items():
                sel = s['lab_id']==lab
                for key in ('past_stress_cmd','stress_hold_cmd'):
                    s[key][sel] = (s[key][sel]*gt['stress_gains'][lab]+gt['stress_biases'][lab]-b)/a
        mask = ~np.isin(base['lab_id'],wing)
        def rows(points, glitch):
            cols = [list(base[k][mask]) for k in keys]
            for lab,(a,b) in truth.items():
                for i,x in enumerate(points):
                    for col,value in zip(cols,(lab,x,a*x+b+(0.16 if i==glitch else 0))):
                        col.append(value)
            return dict(zip(keys,map(np.asarray,cols)))
        archive = m._ARCHIVE/'prior.pkl'
        for name,points,glitch,backed in [('absent',(),-1,False),('spot',(.8,),-1,False),('EMI',(-1.2,.25,1.15),1,False),('archived+EMI',(-1.2,.25,1.15),1,True),('dense+EMI',(-1.8,-.9,0.,.9,1.8),2,False)]:
            if backed:
                with archive.open('wb') as f:
                    pickle.dump([{'dataset':{'calibration_load_cells':rows((-1.8,-.9,0.,.9,1.8),-1)}}],f)
            else:
                archive.unlink(missing_ok=True)
            shifted['calibration_load_cells'] = rows(points,glitch)
            validate(f'{wing} {name}', shifted)
