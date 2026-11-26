import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',p+'alpha_w01_batch_04_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
z=pickle.load(open(p+'public_alpha_w01_batch_04.pkl','rb'))[0]; d=z['dataset']; g=z['gt']
o=s.solve_alpha_w01_batch_04(d)
for ok,gk in [('estimated_taus','taus'),('hereditary_weights','theta'),('viscoplastic_weights','vp_theta')]:
 a,b=o[ok],np.asarray(g[gk]); print(ok,'fit',a.round(6),'truth',b.round(6),'max abs error',np.max(abs(a-b)))
rms=lambda x: np.sqrt(np.mean(np.asarray(x)**2))
st=d['test']; dt=d['past_dt']; t=d['eval_times']; sy=d['yield_stress']; h,hold=s._stress(st,s._calibrate(d['calibration_load_cells'])); ov,w=s._plastic(h,hold,dt,sy)
missing=~np.isfinite(st['coupon_features_raw']).all(1)
e=o['predicted_strain']-g['te_clean']; r=o['initial_rate_pred']-g['te_true_rate0']; print('public total curve/rate RMS',rms(e),rms(r),'first/last curve RMS',rms(e[:,0]),rms(e[:,-1]))
for key,m in [(f'lab {i}',st['lab_id']==i) for i in range(4)]+[('complete',~missing),('missing',missing),('hold only',(ov!=0)&(w==0)),('history only',(ov==0)&(w!=0)),('hold and history',(ov!=0)&(w!=0)),('neither',(ov==0)&(w==0))]:
 if m.any(): print(key,m.sum(),'curve/rate RMS',rms(e[m]),rms(r[m]))
print('test onset observation noise RMS',rms(st['instant_rate_obs']-g['te_true_rate0']))
tr=d['train']; lab=tr['lab_id']; fullx,_,complete,_=s._align(tr,st)
for j in range(4):
 a=lab!=j; b=~a
 ds=dict(d,train={k:v[a] for k,v in tr.items()},test={k:v[b] for k,v in tr.items() if k!='strain_curves_obs'})
 # Preserve the original reference-panel population by passing the original test
 # coupons as unlabeled controls; only measured curves/rates from other labs fit the law.
 maps=s._calibrate(d['calibration_load_cells']); tau,theta=s._fit(ds['train'],fullx[a],complete[a],maps,dt,t,sy)
 hist,hold=s._stress(ds['test'],maps); A,B=s._design(tau,hist,hold,fullx[b],dt,t,*s._plastic(hist,hold,dt,sy))
 rate=B@theta; delta=np.where(~complete[b],tr['instant_rate_obs'][b]-rate,0.)
 pred=tr['strain_0'][b,None]+A@theta+delta[:,None]*t
 print('OOF lab',j,'curve RMS',rms(pred-tr['strain_curves_obs'][b]),'complete rate RMS',rms((rate-tr['instant_rate_obs'][b])[complete[b]]),'taus',tau.round(5),'plastic',theta[7:].round(5))
