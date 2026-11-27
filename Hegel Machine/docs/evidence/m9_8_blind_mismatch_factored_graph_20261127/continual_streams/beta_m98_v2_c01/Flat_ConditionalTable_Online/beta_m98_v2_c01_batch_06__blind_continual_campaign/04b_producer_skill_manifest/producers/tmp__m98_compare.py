import glob, importlib.util, pickle
import numpy as np
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = item['study']
z = s['Z_nuisance']
mods = {}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
    k=f.split('_')[-2]
    spec=importlib.util.spec_from_file_location('m'+k,f)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[k]=m
for ck in ('s00','01','03','04'):
    yc = [mods[ck]._calibrate(dict(s,y_obs_raw=p['y_pulse_raw'])) for p in item['campaign_diagnostics']]
    for fk in ('s00','01','03','04'):
        S,U=mods[fk]._forward(s)
        vals=[]
        for p,y in zip(item['campaign_diagnostics'],yc):
            e=S@p['w_pulse']-y
            ep=e-z@np.linalg.lstsq(z,e,rcond=None)[0]
            vals.append((round(np.sqrt(np.mean(e**2)),8),np.round(np.sqrt(np.mean(ep.reshape(3,-1)**2,axis=1)),8).tolist()))
        print('cal',ck,'wall',fk,vals)
print('observation max',np.max(np.abs(s['y_obs_raw'])))
