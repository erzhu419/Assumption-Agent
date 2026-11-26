import importlib.util, pickle, sys, time
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_w03_batch_07_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w03_batch_07.pkl','rb'))[0]['dataset']; tr=d['train']; t=d['eval_times']; labels=tr['lab_id']; miss=np.isnan(tr['coupon_features_raw']).any(1)
print('calibration',m._calibrate(d['calibration_load_cells'],labels))
def subset(b,sel,drop=()): return {k:v[sel] for k,v in b.items() if k not in drop}
curves=np.empty_like(tr['strain_curves_obs']); rates=np.empty_like(tr['instant_rate_obs'])
for lab in np.unique(labels):
    val=labels==lab
    # Keep the reference panel fixed by retaining original test covariates.
    test={k:np.concatenate((v[val],d['test'][k])) for k,v in tr.items() if k!='strain_curves_obs'}
    fold=dict(d,train=subset(tr,~val),test=test)
    out=m.solve_alpha_w03_batch_07(fold); curves[val]=out['predicted_strain'][:val.sum()]; rates[val]=out['initial_rate_pred'][:val.sum()]
    print('fold',lab,np.round(out['estimated_taus'],6),np.round(out['hereditary_weights'],6),out['viscoplastic_weights'])
ec=curves-tr['strain_curves_obs']; er=rates-tr['instant_rate_obs']; x,h,p=m._prep(d)
def report(name,s):
    print(name,s.sum(),'curve/rate rmse',np.round([np.sqrt(np.mean(ec[s]**2)),np.sqrt(np.mean(er[s]**2))],6),'p90 abs',np.round([np.quantile(abs(ec[s]),.9),np.quantile(abs(er[s]),.9)],6),'median curve by time',np.round(np.median(ec[s],axis=0),5))
for lab in np.unique(labels): report(f'lab {lab}',labels==lab)
report('missing',miss); report('complete',~miss)
report('high |hold|',abs(h[:len(labels)])>=np.quantile(abs(h[:len(labels)]),.85)); report('large stress reversal',np.sign(h[:len(labels)])!=np.sign(p[:len(labels),-1]));
a=time.monotonic(); out=m.solve_alpha_w03_batch_07(d); print('runtime s',round(time.monotonic()-a,3));print('full params',out['estimated_taus'],out['hereditary_weights'],out['viscoplastic_weights']); print('output shapes',{k:v.shape for k,v in out.items()})
