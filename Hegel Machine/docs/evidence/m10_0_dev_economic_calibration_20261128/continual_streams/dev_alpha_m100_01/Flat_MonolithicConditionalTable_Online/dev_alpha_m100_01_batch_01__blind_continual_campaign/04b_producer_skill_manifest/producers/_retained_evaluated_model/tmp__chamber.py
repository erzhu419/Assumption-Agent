import pickle,sys,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_alpha_m100_01_s00_solver import _features,_bridge,_loads,_fit,_design
np.set_printoptions(precision=5,suppress=True,linewidth=140)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te,c=d['train'],d['test'],i['campaign_diagnostics']; n=len(tr['lab_id']); dt,t=d['past_dt'],d['eval_times']; cal=d['calibration_load_cells']; b=_bridge(cal)
R,Y=cal['bridge_raw'],cal['true_channel_loads']; print('bridge',b); print('train hold history loads ranges',[(v.min(),v.max()) for v in _loads(tr,b)]); print('diag hold history loads ranges',[(v.min(),v.max()) for v in _loads(c,b)]);
from itertools import combinations_with_replacement
for p in range(1,4):
 A=np.column_stack([np.ones(len(R))]+[np.prod(R[:,k],axis=1) for q in range(1,p+1) for k in combinations_with_replacement(range(4),q)]); pinv=np.linalg.pinv(A); E=Y-A@(pinv@Y); E=E/(1-np.einsum('ij,ji->i',A,pinv))[:,None]; print('cal LOO degree',p,'channel rms',np.mean(E**2,axis=0)**.5,'total rms',np.mean(E.sum(axis=1)**2)**.5)
s=cal['single_channel_sweeps'];print('sweeps:');
for lab in np.unique(s['lab_id']):
 m=s['lab_id']==lab; x0,y0=s['true_stress'][m],s['cmd_stress'][m]; print(lab, np.array([x0,y0]).T[::2], [np.std(y0-np.polyval(np.polyfit(x0,y0,p),x0)) for p in range(1,5)])
x=_features(tr,te); yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; ta,th=_fit(*_loads(tr,b),x[:n],dt,t,yc,yr); print('taus theta',ta,th)
xc=_features(dict((k,tr[k]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')),c)[n:]; past,hold=_loads(c,b); cy=c['strain_curves_obs']-c['strain_0'][:,None]; cr=c['instant_rate_obs'];
def metrics(c,r,y,z):
 e=c-y; f=r-z;return 'curve/rate MAE',np.mean(abs(e)),np.mean(abs(f)),'curve/rate median',np.median(abs(e)),np.median(abs(f)),'90%',np.quantile(abs(e),.9),np.quantile(abs(f),.9)
from scipy.optimize import least_squares
Q=np.vstack((np.column_stack((-np.expm1(-t[:,None]/ta),t)),np.r_[1/ta,1])); Y=np.c_[cy,cr]; U=Y@np.linalg.pinv(Q.T)
D=np.c_[np.ones(len(hold)),hold,past]; L=np.eye(4)[c['lab_id']]; S=np.eye(3)[c['stratum']]; F=c['coupon_features_raw'];
for k in range(3):
 print('amplitude',k,'std',np.std(U[:,k]))
 for name,m in [('physical',D),('features',np.c_[np.ones(len(X)) if False else D[:,0],xc]),('both',np.c_[D,xc]),('bylab',np.c_[D,xc,L[:,1:]]),('bystrata',np.c_[D,xc,S[:,1:]])]:
  pinv=np.linalg.pinv(m); beta=pinv@U[:,k]; pred=m@beta; loo=(U[:,k]-pred)/(1-np.einsum('ij,ji->i',m,pinv)); print(name, 'beta',beta,'training rms',np.std(U[:,k]-pred),'loo rms',np.mean(loo**2)**.5)
print('amplitudes / theta', U / np.r_[th[:2],1.]); print('diag baseline intrinsic drift', th[2]+xc@th[3:]);
# Compare plausible reference populations while holding chamber physical basis fixed.
def custom(parts,refset):
 labs=np.concatenate([a['lab_id'] for a in parts]); x=np.concatenate([a['coupon_features_raw'] for a in parts]);
 for j in range(4):
  p=np.concatenate([a['ref_coupons'][a['lab_id']==j] for a in refset]); x[labs==j]=(x[labs==j]-np.median(p.mean(0)))/np.median(p.std(0));
 return x
for name,refs in [('c',[c]),('trte',[tr,te]),('trc',[tr,c]),('all',[tr,te,c])]:
 X=custom([c],refs); ok=np.all(np.isfinite(X),axis=1);
 for form in ['drift linear','drift full']:
  E=np.c_[D[:,0],X] if form=='drift linear' else np.c_[D,X]
  E=E[ok]; coef=np.linalg.lstsq(E,U[ok,2],rcond=None)[0]; print(name,form,'std',np.std(E@coef-U[ok,2]),'beta',coef)
