import importlib.util,pickle,numpy as np
from sklearn.model_selection import StratifiedKFold
p='/holdout_task/alpha_w03_batch_05_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_alpha_w03_batch_05.pkl','rb'))[0]; d=a['dataset']; tr=d['train']; n=len(tr['lab_id']); x,h,p=m._prep(d); x,h,p=x[:n],h[:n],p[:n]; dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; labs=tr['lab_id']; strata=tr['stratum']; ok=np.isfinite(tr['coupon_features_raw']).all(1); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; z=tr['instant_rate_obs']; E=np.zeros((3,n,len(t))); R=np.zeros((3,n)); rate_delta=np.zeros(n)
for train,val in StratifiedKFold(n_splits=4,shuffle=True,random_state=29).split(labs,labs*10+strata):
 train=train[ok[train]]; tau,b=m._fit(x[train],h[train],p[train],dt,t,sy,y[train],z[train]); c,r=m._design(tau,x,h,p,dt,t,sy); ec=y-c@b; er=z-r@b
 # Compare a time-linear lab drift estimated on training coupons only against the contractual baseline.
 early=t>=np.median(t); projected=np.median(ec[:,early]/t[early],axis=1)
 shift={l:np.median(projected[train[labs[train]==l]]) for l in np.unique(labs)}; vshift=np.array([shift[l] for l in labs[val]]); missing=~ok[val]; rate_delta[val]=er[val]
 corr0=np.where(missing,er[val],0.); corr1=np.where(missing,er[val],vshift); corr2=np.where(missing,er[val],(vshift+er[val])/2)
 for i,corr in enumerate((corr0,corr1,corr2)):
  E[i,val]=ec[val]-corr[:,None]*t; R[i,val]=er[val]-corr
 print('training lab drift estimates',np.round(list(shift.values()),6))
# A common outlier mask keeps the comparisons on identical measurements.
valid=np.abs(E[0])<.2
for lab in np.unique(labs):
 for st in np.unique(strata[labs==lab]):
  v=(labs==lab)&(strata==st)&ok; mask=v[:,None]&valid
  print('held cohort',int(lab),int(st),int(v.sum()),'strain',np.round(np.sqrt(np.mean(E[:,mask]**2,axis=1)),6),'rate model baseline/drift',np.round(np.sqrt(np.mean(R[:2,v]**2,axis=1)),6))
for missing in (False,True):
 v=ok!=missing; mask=v[:,None]&valid; print('missing',missing,'n',v.sum(),'curve RMSE',np.round(np.sqrt(np.mean(E[:,mask]**2,axis=1)),6))
