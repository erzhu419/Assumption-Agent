import importlib.util
import pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w01_s00_solver.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_alpha_w01_s00.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr=d['train']; labs=tr['lab_id']; dt=d['past_dt']; t=d['eval_times']; y=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; cal=m._calibrate(d['calibration_load_cells']); n=len(labs)
for layout in (False, True):
    x=m._align(d,layout)[:n]; inp=m._inputs(tr,x,cal)
    print('feature_major',layout)
    for lab in range(4):
        fit=labs!=lab; val=~fit
        law=lambda mask:m._Law(tuple(k[mask] for k in inp),dt,t)
        p=m._fit(law(fit), y[fit], rate[fit]); c,r=law(val).evaluate(p)
        print(lab,'tau',np.round(p[:2],5),'curve median/p90',np.round(np.quantile(abs(c-y[val]),[.5,.9]),6),'rate median/p90',np.round(np.quantile(abs(r-rate[val]),[.5,.9]),6), 'horizon p90 max', np.quantile(abs(c-y[val]),.9,axis=0).max())
    p=m._fit(m._Law(inp,dt,t),y,rate); print('full',p)
