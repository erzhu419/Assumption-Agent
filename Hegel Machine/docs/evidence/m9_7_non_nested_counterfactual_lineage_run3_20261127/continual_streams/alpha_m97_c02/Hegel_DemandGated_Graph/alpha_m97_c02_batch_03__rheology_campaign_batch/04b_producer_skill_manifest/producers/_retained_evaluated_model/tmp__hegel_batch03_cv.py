import importlib.util, pickle
import numpy as np
base = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', base + 'alpha_m97_c02_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(base+'public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
tr, te = d['train'], d['test']; dt, times = d['past_dt'], d['eval_times']
lab = tr['lab_id']; cal = m._calibrate(d['calibration_load_cells']); print('calibration',cal.tolist())
x = m._align(tr,te)[:len(lab)]; p,h = m._physical(tr,cal)
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']; er_y = np.empty_like(y); er_r = np.empty_like(r)
rmse = lambda v: float(np.sqrt(np.mean(v*v)))
for l in [-1,0,1,2,3]:
    fit = lab != l; pred = ~fit if l >= 0 else fit
    taus,w = m._fit(p[fit],h[fit],x[fit],dt,times,y[fit],r[fit]); a,b = m._design(p[pred],h[pred],x[pred],dt,times,taus)
    dy,dr = a@w-y[pred], b@w-r[pred]
    if l >= 0: er_y[pred],er_r[pred] = dy,dr
    print(l, 'taus',taus,'weights',w,'rmse',rmse(dy),rmse(dr),'abs residual 50/95/100%',np.quantile(abs(dy),[.5,.95,1]),np.quantile(abs(dr),[.5,.95,1]))
for name,mask in [('missing',np.isnan(tr['coupon_features_raw']).any(1)),('large_stress',abs(h)>=np.quantile(abs(h),.9)),('negative',h<0),('positive',h>=0)]:
    print(name,mask.sum(),rmse(er_y[mask]),rmse(er_r[mask]))
print('OOF curve and rate median absolute errors',np.median(abs(er_y)),np.median(abs(er_r)))
print('time-wise OOF strain RMSE',np.sqrt(np.mean(er_y**2,axis=0)))
