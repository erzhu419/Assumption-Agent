import importlib.util,pickle,numpy as np
p='/holdout_task/alpha_m96_w03_batch_03_solver.py'; s=importlib.util.spec_from_file_location('b03',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_alpha_m96_w03_batch_03.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; x,xe=m._align(tr,te); cal=m._calibrate(d['calibration_load_cells']); h=m._Hold(d['past_dt'],d['eval_times'],tr,x,cal)
y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]; np.set_printoptions(precision=4,suppress=True)
p=m._fit(h,tr); print('full parameters',p); a,_=h.design(p[:2]); r=y-a@p[2:]; print('per-channel median abs',np.nanmedian(abs(r),0)); print('per-channel q90 abs',np.nanquantile(abs(r),.9,axis=0))
print('max residual rows',[(int(i),int(tr['lab_id'][i]),np.round(r[i],3).tolist()) for i in np.argsort(np.nanmax(abs(r),axis=1))[-5:]])
for label in [0,1,2,3]:
 use=tr['lab_id']==label; p=m._fit(h,tr,~use); a,_=h.design(p[:2]); rr=y[use]-a[use]@p[2:]; print('LOLO',label,'taus',p[:2],'curve rmse',np.sqrt(np.nanmean(rr[:,:-1]**2)),'horizon RMSE',np.sqrt(np.nanmean(rr**2,axis=0)),'curve MAE',np.nanmean(abs(rr[:,:-1])))
for wing in [0,1]:
 use=tr['lab_id']//2==wing; p=m._fit(h,tr,~use); a,_=h.design(p[:2]); rr=y[use]-a[use]@p[2:]; print('LOWO',wing,'taus',p[:2],'curve rmse',np.sqrt(np.nanmean(rr[:,:-1]**2)),'horizon RMSE',np.sqrt(np.nanmean(rr**2,axis=0)))
print('reference scale by lab',[(i,np.median(np.nanstd(np.vstack((tr['ref_coupons'],te['ref_coupons']))[np.r_[tr['lab_id'],te['lab_id']]==i],0))) for i in range(4)])
