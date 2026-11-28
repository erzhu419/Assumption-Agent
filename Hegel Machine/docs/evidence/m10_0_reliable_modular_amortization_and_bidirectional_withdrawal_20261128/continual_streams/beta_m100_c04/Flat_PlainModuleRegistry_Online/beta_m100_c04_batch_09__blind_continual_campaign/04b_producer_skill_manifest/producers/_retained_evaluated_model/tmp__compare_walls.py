import itertools, pickle
import numpy as np
from batch_solver import _up, _PAY
from downstream_solver import _forward, _fit
for item in pickle.load(open('/holdout_task/public_batch.pkl','rb')):
    s = _up.calibrate_upstream(item['study'])
    y,z = s['y_obs_raw'],s['Z_nuisance']
    n = len(s['sensor_ij'])
    assert n % 4 == 0
    ids = np.arange(len(y)) % n // 4
    r_all = []
    for walls in itertools.product(('robin','sealed','periodic'), repeat=2):
        a,_ = _forward(s,walls)
        w,nu,b,t,gamma = _fit(s,a,y,z)
        residual = a@w+z@gamma-y
        errors = []
        for group in np.unique(ids):
            keep = ids != group
            wf,_,_,_,gf = _fit(s,a[keep],y[keep],z[keep])
            errors.append(a[~keep]@wf+z[~keep]@gf-y[~keep])
        errors=np.array(errors)
        rmse=lambda x: np.sqrt(np.mean(x*x))
        r_all.append((rmse(errors),max(map(rmse,errors)),walls,rmse(residual),w.round(4).tolist()))
    for r in sorted(r_all): print(r)
