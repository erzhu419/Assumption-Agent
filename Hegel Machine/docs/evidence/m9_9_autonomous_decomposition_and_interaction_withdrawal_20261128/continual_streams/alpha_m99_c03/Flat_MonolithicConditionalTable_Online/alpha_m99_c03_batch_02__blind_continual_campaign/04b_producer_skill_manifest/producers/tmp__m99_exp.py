import pickle, importlib.util, numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=130)
s=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/alpha_m99_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=a['dataset']; tr=ds['train']; te=ds['test']; d=a['campaign_diagnostics']; dt=ds['past_dt']; t=ds['eval_times']; cal=ds['calibration_load_cells']; n=len(tr['lab_id']); x=m._features(tr,te); ny=len(x); joint={k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; xd=m._features(joint,d)[ny:]; xd2=m._features(tr,d)[n:]; bridge=m._calibration(cal); raw=cal['bridge_raw']; truth=cal['true_channel_loads']; pr=np.c_[np.ones(len(raw)),raw]@bridge; print('cal ranges',raw.min(0),raw.max(0),'residual rms and max',np.sqrt(np.mean((pr-truth)**2,axis=0)),np.max(abs(pr-truth),axis=0),'axial rms',np.std((pr-truth).sum(1))); print('sweeps',[(k,type(v).__name__,list(v) if isinstance(v,dict) else np.shape(v)) for k,v in cal['single_channel_sweeps'].items()]); print('dt,t',dt,t,'diagnostic feature method difference',np.max(abs(xd-xd2),axis=0)); print('ref moment',[(g,[(int(np.sum(b['lab_id']==g)),np.mean(b['ref_coupons'][b['lab_id']==g]) if np.any(b['lab_id']==g) else None) for b in (tr,te,d)]) for g in np.unique(joint['lab_id'])]);
Ytr=tr['strain_curves_obs']-tr['strain_0'][:,None]; Rtr=tr['instant_rate_obs']; train=m._coupon_inputs(tr,x[:n],bridge); taus,th=m._fit(train,dt,t,Ytr,Rtr); print('tau theta',taus,th); C,R=m._design(taus,train,dt,t); print('train errors [mad rmse max]',[(np.median(abs(e)),np.sqrt(np.mean(e**2)),np.max(abs(e))) for e in (C@th-Ytr,R@th-Rtr)]);
p,h,Z=m._coupon_inputs(d,xd,bridge); F=Z@th[2:]; Y=d['strain_curves_obs']-d['strain_0'][:,None]; r=d['instant_rate_obs']; C,R=m._design(taus,(p,h,Z),dt,t); P0,Q0=m._chamber(taus,th,h,xd,t); B=-np.expm1(-t[:,None]/taus); z=h[:,None]-R[:,:2]*taus; print('reference ranks/svs',np.linalg.svd(Y,compute_uv=False)); print('bench / reset residual [median abs, rms, max]',[(np.median(abs(e)),np.sqrt(np.mean(e**2)),np.max(abs(e))) for e in (C@th-Ytr if False else C@th-Y,R@th-r,P0-Y,Q0-r)])
from scipy.optimize import minimize_scalar
Y0=Y-C[:,:,:2]@th[:2]; r0=r-R[:,:2]@th[:2]
def rms(v): return np.sqrt(np.mean(v*v))
print('DRIFT-ONLY SHAPE TEST; curve RMS with observed initial drift, then free amplitudes; rate RMS from curves alone')
for name,g in [('linear',t),('log',np.log1p(t)),('half log',2*np.log1p(t/2)),('double log',np.log1p(2*t)/2),('sqrt',2*(np.sqrt(1+t)-1)),('exp',-np.expm1(-t)),('rational',t/(1+t))]:
 q=Y0@g/(g@g); print(name,np.round([rms(Y0-r0[:,None]*g),rms(Y0-q[:,None]*g),rms(q-r0)],6))
for name,fun in [('log',lambda k: np.log1p(k*t)/k),('exp',lambda k: -np.expm1(-k*t)/k),('rational',lambda k:t/(1+k*t)),('sqrt',lambda k: 2*t/(np.sqrt(1+k*t)+1))]:
 def err(k):
  g=fun(k); q=(Y0@g+0.8*r0)/(g@g+0.8); return rms(Y0-q[:,None]*g)**2+0.8/len(t)*rms(q-r0)**2
 v=minimize_scalar(err,bounds=(0.0001,15),method='bounded'); print('parametric',name,v.x,np.sqrt(v.fun))
print('drift-only singular values',np.linalg.svd(Y0,compute_uv=False)); print('diagnostic drift/rate [observed initial drift, intrinsic, drift projected on log]',np.round(np.c_[r0,F,Y0@np.log1p(t)/sum(np.log1p(t)**2)],4))
print('BASE+SHAPE curve correction amplitude model; in-sample rms, leave-one-out rms')
xxs={'joint':xd,'train+diag':xd2,'diag_only':m._features(d,d)[:len(h)]}
ref=joint['ref_coupons']; labs=joint['lab_id']; dx=d['coupon_features_raw'].copy(); wi=np.unique(np.c_[joint['lab_id'],joint['stratum']],axis=0,return_inverse=True,return_counts=True); weights=1/wi[2][wi[1]]; avg=np.array([np.average(x[np.isfinite(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']])[:,j]),j],weights=weights[np.isfinite(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']])[:,j])]) for j in range(4)])
for lab in np.unique(labs):
 panel=ref[labs==lab]; g=d['lab_id']==lab; dx[g]=(dx[g]-np.median(panel.mean(0)))/np.median(panel.std(0))
xxs['frozen_joint']=np.where(np.isfinite(dx),dx,avg)
from itertools import combinations_with_replacement
G=-np.expm1(-t); q=(Y0@G+r0)/(G@G+1)
y0=C[:,:,:2]@th[:2]; v0=R[:,:2]@th[:2]
for name,xx in xxs.items():
 f=th[2]+xx@th[3:]; N=len(f); sets={'tied':np.c_[np.ones(N),f], 'features':np.c_[np.ones(N),xx], 'joint_stress':np.c_[np.ones(N),f,f*h], 'drift_squared':np.c_[np.ones(N),f,f*f], 'sum_gap':np.c_[np.ones(N),f,f*(h-z.mean(1))]}; gs=np.column_stack([f,xx,h,z]); labels=['f','x0','x1','x2','x3','hold','z0','z1'];
 for i,j in combinations_with_replacement(range(gs.shape[1]),2): sets[labels[i]+'*'+labels[j]]=np.c_[np.ones(N),f,gs[:,i]*gs[:,j]]
 out=[]
 for tag,A in sets.items():
  beta=np.linalg.lstsq(A,q,rcond=None)[0]; Q=np.linalg.qr(A,mode='reduced')[0]; qloo=q-(q-A@beta)/(1-np.sum(Q**2,axis=1)); E=y0+qloo[:,None]*G-Y; er=v0+qloo-r
  out.append((rms(E),rms(er),tag,beta.round(5).tolist(),max(np.sqrt(np.mean(E**2,axis=1))),max(abs(er))))
 print('EXP',name,'unscaled rms',rms(q-f),'baseline and winners', [r for r in out if r[2] in ('tied','features')], sorted(out)[:4]);
 for lab in np.unique(d['lab_id']):
  ids=d['lab_id']==lab; A=sets['tied']; pred=A[ids]@np.linalg.lstsq(A[~ids],q[~ids],rcond=None)[0]; print('LAB',lab,rms(y0[ids]+pred[:,None]*G-Y[ids]),rms(v0[ids]+pred-r[ids]))
print('diag-only fitted',np.c_[d['lab_id'], np.any(np.isnan(d['coupon_features_raw']),axis=1), q, th[2]+xxs['diag_only']@th[3:]])
for b in (tr,te):
 u=b.get('aux_chamber_telemetry'); print('AUX',np.min(u,axis=0),np.max(u,axis=0),np.std(u,axis=0))
