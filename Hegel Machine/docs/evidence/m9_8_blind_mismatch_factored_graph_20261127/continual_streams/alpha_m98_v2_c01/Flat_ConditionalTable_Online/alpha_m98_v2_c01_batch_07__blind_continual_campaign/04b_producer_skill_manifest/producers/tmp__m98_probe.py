import importlib.util,pickle,numpy as np
from pathlib import Path
p=Path('/tmp/hegel_continual_archive'); item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; tr,te,diag=d['train'],d['test'],item['campaign_diagnostics']; cal=d['calibration_load_cells']; dt,t=d['past_dt'],d['eval_times']; targets=(diag['strain_curves_obs']-diag['strain_0'][:,None],diag['instant_rate_obs']); raw=cal['bridge_raw']; loads=cal['true_channel_loads']; print('raw range',np.quantile(raw,[0,.02,.5,.98,1],axis=0)); print('sweeps',[(k,getattr(v,'shape',list(v) if isinstance(v,dict) else None)) for k,v in cal['single_channel_sweeps'].items()]); print('target std',*[np.std(y) for y in targets]); mods={}
for n in ('01','02','03','06'):
 s=importlib.util.spec_from_file_location('m'+n,next(p.glob('*batch_'+n+'_solver.py'))); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); mods[n]=m
 local=dict(d,test=diag); o=m.solve_batch(local); curves=o['predicted_strain']-diag['strain_0'][:,None]; rate=o['initial_rate_pred']; print(n,'tau',o['estimated_taus'],'theta',o['hereditary_weights'],'RMSE',*[np.sqrt(np.mean((a-b)**2)) for a,b in zip((curves,rate),targets)],'lab RMSE',[[round(np.sqrt(np.mean((a[diag['lab_id']==j]-b[diag['lab_id']==j])**2)),5) for j in np.unique(diag['lab_id'])] for a,b in zip((curves,rate),targets)])
clean=np.max(np.abs(raw-np.median(raw,axis=0)),axis=1)<8
for n in ('02','06'):
 m=mods[n]; bridge=m._axial_load_map(cal); spec=m._specimen(dict(past_channel_raw=raw[:,None,:],hold_channel_raw=raw),np.zeros((len(raw),4)),bridge); print('cal error',n,np.quantile(np.abs(spec[1][clean]-loads.sum(axis=1)[clean]),[.5,.9,1])); print('map',bridge)
