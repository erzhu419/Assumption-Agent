import importlib.util, pickle
import numpy as np
path='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',path+'alpha_m97_c02_batch_03_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.load(open(path+'public_alpha_m97_c02_batch_03.pkl','rb'))[0]['dataset']
f=m.solve_alpha_m97_c02_batch_03
r=f(d)
print('full parameters',*[np.round(r[k],7) for k in ('estimated_taus','hereditary_weights','regime_mode_weights')]); print('calibration',m._calibration(d))
tr=d['train']; fields=d['test'].keys(); e=[]; er=[]; spread=[]
for lab in range(4):
 v=tr['lab_id']==lab
 fold=dict(d,train={k:x[~v] for k,x in tr.items()},test={k:np.concatenate((tr[k][v],d['test'][k])) for k in fields})
 p=f(fold); n=v.sum(); ec=p['predicted_strain'][:n]-tr['strain_curves_obs'][v]; rr=p['initial_rate_pred'][:n]-tr['instant_rate_obs'][v]
 e.append(ec); er.append(rr); spread.append(p['estimated_taus']); missing=np.isnan(tr['coupon_features_raw'][v]).any(axis=1)
 print('lab',lab,'taus',p['estimated_taus'],'strain MAE/p95/max',np.round(np.percentile(np.abs(ec),[95,100]),6),np.mean(np.abs(ec)),'rate MAE/p95/max',np.round(np.percentile(np.abs(rr),[95,100]),6),np.mean(np.abs(rr)), 'complete and imputed MAE',[(int(g.sum()),np.mean(np.abs(ec[g])),np.mean(np.abs(rr[g]))) for g in (~missing,missing) if g.any()]); print('strain abs by time',np.round(np.mean(np.abs(ec),axis=0),6))
print('tau range',np.min(spread,axis=0),np.max(spread,axis=0))
print('archived per-sweep calibration')
import json
for q in json.load(open(m._HISTORY)):
 l,x,y=map(np.asarray,(q['lab_id'],q['cmd_stress'],q['true_stress'])); print(q['episode_short_name'],[m._line(x[l==i],y[l==i]).round(5).tolist() for i in range(4)])
