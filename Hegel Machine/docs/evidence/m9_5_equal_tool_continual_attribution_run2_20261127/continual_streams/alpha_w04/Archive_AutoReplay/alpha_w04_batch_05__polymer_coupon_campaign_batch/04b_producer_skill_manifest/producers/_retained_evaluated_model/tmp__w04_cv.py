import importlib.util,pickle,sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
path='/holdout_task/alpha_w04_batch_05_solver.py' if len(sys.argv)<2 else sys.argv[1]
spec=importlib.util.spec_from_file_location('solver',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_w04_batch_05.pkl','rb'))[0]['dataset']
tr=d['train']; sol=m.solve_alpha_w04_batch_05
print('calibration',m._calibrate(d['calibration_load_cells'])); out=sol(d)
print('fit',out['estimated_taus'],out['hereditary_weights'],out['viscoplastic_weights'])
s=np.empty_like(tr['strain_curves_obs']); r=np.empty_like(tr['instant_rate_obs']); params=[]
folds=StratifiedKFold(5,shuffle=True,random_state=94).split(tr['lab_id'],tr['lab_id'])
for fit,val in folds:
    dd={**d,'train':{k:v[fit] for k,v in tr.items()},'test':{k:v[val] for k,v in tr.items() if k!='strain_curves_obs'}}
    o=sol(dd); s[val]=o['predicted_strain']; r[val]=o['initial_rate_pred']; params.append(np.r_[o['estimated_taus'],o['viscoplastic_weights']])
hist,hold=m._stress(tr,m._calibrate(d['calibration_load_cells'])); missing=np.isnan(tr['coupon_features_raw']).any(axis=1); above=(np.abs(hold)>d['yield_stress'])|(np.abs(hist)>d['yield_stress']).any(axis=1)
err_s=s-tr['strain_curves_obs']; err_r=r-tr['instant_rate_obs']; print('fold params',np.round(params,5))
for name,idx in [('all',np.ones(len(s),bool))]+[(f'lab{c}',tr['lab_id']==c) for c in range(4)]+[('below',~above),('above',above),('missing',missing),('complete',~missing)]:
    if idx.any():
        a=err_s[idx]; b=err_r[idx]; print(name,len(a),'RMSE',np.sqrt(np.mean(a*a)),np.sqrt(np.mean(b*b)),'median abs',np.median(abs(a)),np.median(abs(b)),'bias',np.median(a),np.median(b))
