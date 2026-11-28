import pickle
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
import upstream_transducer as u

alternatives = ('paired', 'shared ratios', 'rational correction', 'sum/difference correction')
for item in pickle.load(open('/holdout_task/public_batch.pkl', 'rb')):
    c = item['dataset']['calibration_load_cells']
    t, r = c['true_channel_loads'], c['bridge_raw']
    sw = c['single_channel_sweeps']
    # The preceding public run found gross spikes; report the excluded magnitudes
    # using a screen independent of the predictions being compared here.
    clean = np.max(np.abs(r), axis=1) < 5
    clean_s = np.abs(sw['cmd_stress']) < 5
    print('max raw magnitude retained/rejected (rows and sweeps)', np.max(np.abs(r[clean])), np.max(np.abs(r[~clean]),axis=1), np.max(np.abs(sw['cmd_stress'][clean_s])), np.abs(sw['cmd_stress'][~clean_s]))
    features = lambda x: u._design(x, x[..., u._PARTNER])
    def forward(x, coef):
        return np.sum(features(x) * coef, axis=-1)
    def extra(x, kind):
        p = x[..., u._PARTNER]
        if kind == 'rational correction':
            return np.stack((x/(1+x*x), p*p/(1+p*p), x*p/(1+(x-p)**2)), axis=-1)
        return np.stack((np.sin(2.5*(x+p))-np.sin(2.5*x)-np.sin(2.5*p),
                         np.sin(2.5*(x-p))-np.sin(2.5*x)+np.sin(2.5*p), x*p),axis=-1)
    for kind in alternatives:
        err = np.full_like(t, np.nan)
        for f in range(5):
            train = np.arange(len(t))%5 != f
            s_train = np.arange(len(clean_s))%5 != f
            coef = u._fit_bridge(dict(c, true_channel_loads=t[train], bridge_raw=r[train], single_channel_sweeps={k:v[s_train] for k,v in sw.items()}))
            sel = ~train & clean
            mix = train & clean
            s = s_train & clean_s
            j = sw['lab_id'][s]
            v = sw['true_stress'][s]
            y = sw['cmd_stress'][s]
            sweep_x = np.eye(4)[j] * v[:,None]
            if kind == 'shared ratios':
                def unpack(p):
                    return np.c_[p[:4], p[4:8,None]*np.r_[1,p[8:]][None,:]]
                def shared_res(p):
                    b = unpack(p)
                    return np.r_[(forward(t[mix],b)-r[mix]).ravel(),forward(sweep_x,b)[np.arange(len(j)),j]-y]
                p = np.r_[coef[:,0],coef[:,1],np.median(coef[:,2:]/coef[:,1,None],axis=0)]
                coef = unpack(least_squares(shared_res,p).x)
            if kind in alternatives[2:]:
                a = np.concatenate((features(t),extra(t,kind)),axis=-1)
                b = np.concatenate((features(sweep_x),extra(sweep_x,kind)),axis=-1)
                w = np.array([np.linalg.lstsq(np.r_[a[mix,k],b[j==k,k]],np.r_[r[mix,k],y[j==k]],rcond=None)[0] for k in range(4)])
                guess = u._invert(r[sel],coef)
                vhat = [least_squares(lambda x: np.sum(np.concatenate((features(x),extra(x,kind)),axis=-1)*w,axis=-1)-z, x0,gtol=1e-13).x for z,x0 in zip(r[sel],guess)]
            else:
                vhat = u._invert(r[sel],coef)
            err[sel] = vhat-t[sel]
        inner = np.count_nonzero(t,axis=1)>1
        edge = np.max(np.abs(t),axis=1)>=1.45
        print(kind, 'OOF max/axis/mixed/edge/RMSE', *(float(np.nanmax(np.abs(z))) for z in (err,err[~inner],err[inner],err[edge])),float(np.sqrt(np.nanmean(err**2))))
    print('schema keys (no classification target or relay state)',list(item['dataset']),list(item['dataset']['train']))
