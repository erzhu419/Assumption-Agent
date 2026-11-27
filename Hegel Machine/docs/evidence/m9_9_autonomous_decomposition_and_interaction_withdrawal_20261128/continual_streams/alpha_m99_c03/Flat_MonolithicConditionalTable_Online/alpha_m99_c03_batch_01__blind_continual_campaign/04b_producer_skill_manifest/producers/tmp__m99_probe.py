from m99_inputs import *
from scipy.optimize import least_squares
for label,h,ci in [('train',tr,c),('diag',g,cg)]:
 a,r=m._design(tau,ci,dt,t); ec=h['strain_curves_obs']-h['strain_0'][:,None]-a@th; eR=h['instant_rate_obs']-r@th
 print(label,'curve/rate RMSE',np.sqrt(np.mean(ec**2)),np.sqrt(np.mean(eR**2)),'absolute medians',np.median(abs(ec)),np.median(abs(eR)))
D=np.vstack((E,er)); obs=np.c_[y,yr]; B=np.array([least_squares(lambda b: D@b-o,np.linalg.lstsq(D,o,rcond=None)[0],loss='soft_l1',f_scale=.015).x for o in obs]); print('fixed time basis RMSE/Q50/Q90',np.sqrt(np.mean((B@D.T-obs)**2)),np.quantile(abs(B@D.T-obs),[.5,.9],axis=0).round(4));
p,h,x=cg; nomdr=x@th[2:]; a=R0[:,:2]*tau; z=h[:,None]-a; print('amplitude/hold summary',np.quantile(B[:,:2]/h[:,None],[0,.25,.5,.75,1],axis=0)); print('drift quantiles',np.quantile(B[:,-1],[0,.25,.5,.75,1]),'drift vs baseline',np.linalg.lstsq(np.c_[np.ones(len(h)),nomdr],B[:,-1],rcond=None)[0]);
for name,F in [('hold',h[:,None]),('hold+nominal memory',np.c_[h,z]),('hold+past',np.c_[h,p]),('hold+past+features',np.c_[h,p,x]),('hold+past squared',np.c_[h,p,p*p]),('hold+past+hold*past',np.c_[h,p,h[:,None]*p])]:
 F=np.c_[np.ones(len(h)),F]; co=np.linalg.lstsq(F,B,rcond=None)[0]; rms=[]
 for lab in np.unique(g['lab_id']):
  keep=g['lab_id']!=lab; pred=F[~keep]@np.linalg.lstsq(F[keep],B[keep],rcond=None)[0]; rms.append(np.sqrt(np.mean((pred-B[~keep])**2,axis=0)))
 print(name,'in/LOLO',np.sqrt(np.mean((F@co-B)**2,axis=0)).round(4),np.max(rms,axis=0).round(4));
 if name in ('hold+nominal memory','hold+past'): print('co',co.round(4))
print('bridge matrix',bridge.round(5)); print('first diagnostics observations',obs[:3].round(4)); print('train,test telemetry',[(np.mean(k['aux_chamber_telemetry'],axis=0),np.std(k['aux_chamber_telemetry'],axis=0)) for k in (tr,te)])
