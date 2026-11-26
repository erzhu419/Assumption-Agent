import importlib.util, pickle, sys
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',sys.argv[1] if len(sys.argv)>1 else p+'alpha_ep02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open(p+'public_alpha_ep02.pkl','rb') as f: r=pickle.load(f)[0]
d,g=r['dataset'],r['gt']; o=m.solve_alpha_ep02(d)
rmse=lambda a: np.sqrt(np.mean(np.asarray(a)**2))
for k,j in [('estimated_taus','taus'),('hereditary_weights','theta')]:
 print(k, 'estimated',o[k], 'reference',g[j], 'RMSE',rmse(o[k]-g[j]))
train,test,y,complete=m._prepare(d); labs=d['train']['lab_id']; te=d['test']; pred=test.matrix(np.log(o['estimated_taus']))@o['hereditary_weights']
for is_complete in [True,False]:
 rows=np.isfinite(te['coupon_features_raw']).all(axis=1)==is_complete
 print('test', 'complete' if is_complete else 'imputed',rows.sum(),'rate prior / observed / final RMSE',*[round(rmse(x[rows]-g['te_true_rate0'][rows]),6) for x in [pred[:,-1],te['instant_rate_obs'],o['initial_rate_pred']]])
for lab in np.unique(te['lab_id']):
 rows=te['lab_id']==lab
 print('test lab',lab,'strain / rate RMSE',rmse(o['predicted_strain'][rows]-g['te_clean'][rows]),rmse(o['initial_rate_pred'][rows]-g['te_true_rate0'][rows]))
cal=d['calibration_load_cells']
for lab in np.unique(cal['lab_id']):
 q=cal['lab_id']==lab; x=cal['cmd_stress'][q]; a,b=np.polyfit(x,cal['true_stress'][q],1); print('calibration lab',lab,'resid rms / gain err / bias err',rmse(a*x+b-cal['true_stress'][q]),a-g['stress_gains'][lab],b-g['stress_biases'][lab])
for lab in np.unique(labs):
 tau,coef=m._fit(train,y,complete & (labs!=lab)); q=labs==lab
 pred=train.matrix(np.log(tau))@coef
 upd=m._update_drift(pred, y[:,-1],complete,d['eval_times'])
 print('LOLO lab',lab,'n complete/imputed',sum(q&complete),sum(q&~complete),'taus',np.round(tau,4),'median abs forecast residual prior / updated',*[round(np.median(np.abs(y[q,:-1]-z[q,:-1])),6) for z in [pred,upd]],'prior complete-rate residual MAE',np.median(np.abs(y[q&complete,-1]-pred[q&complete,-1])))
