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
Y=np.c_[cy,cr]; n0=len(Y)
def basis(taus,drift=True):
 q=np.vstack((-np.expm1(-t[:,None]/taus),1/taus)); return np.c_[q,np.r_[t,1.]] if drift else q
for drift in [True,False]:
 def profile(logta):
  q=basis(np.exp(logta),drift); return (Y@np.linalg.pinv(q.T)@q.T-Y).ravel()
 f=least_squares(profile,np.log(ta)); E=profile(f.x).reshape(Y.shape); print('free diagnostic tau',drift,np.exp(f.x),'rms',np.sqrt(np.mean(E**2,axis=0)),'median abs',np.median(abs(E)))
q=basis(ta); U=Y@np.linalg.pinv(q.T); X0=np.r_[np.ones((1,len(xc))),xc.T].T; r=X0@th[2:]; z0=hold[:,None]-_design(ta,past,hold,xc,dt,t)[1][:,:2]*ta
print('lab r gapZ pastLast, offsets scaled by theta'); print(np.c_[c['lab_id'],r,z0,past[:,-1],U[:,:2]/th[:2]-hold[:,None],U[:,2]])
print('corr offset fast vs slow',np.corrcoef((U[:,:2]/th[:2]-hold[:,None]).T))
for refname,refs in [('c',[c]),('trte',[tr,te]),('trc',[tr,c]),('all',[tr,te,c])]:
 X=c['coupon_features_raw'].copy(); labs=c['lab_id']; st=c['stratum']; _,cell,cnt=np.unique(np.c_[labs,st],axis=0,return_inverse=True,return_counts=True); w=1/cnt[cell]
 for j in np.unique(labs):
  pa=np.vstack([p['ref_coupons'][p['lab_id']==j] for p in refs]); X[labs==j]=(X[labs==j]-np.median(pa.mean(0)))/np.median(pa.std(0))
 for imp,fill in [('diagnostic',np.nansum(w[:,None]*X,0)/np.sum(w[:,None]*np.isfinite(X),0)),('train_deploy',None)]:
  Z=np.where(np.isfinite(X),X,fill if fill is not None else _features(tr,te)[np.flatnonzero(np.isnan(tr['coupon_features_raw']).any(1))[0]]) if fill is not None else X.copy()
  if fill is None:
   P=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); T=_features(tr,te); avg=np.array([T[np.flatnonzero(~np.isfinite(P[:,k]))[0],k] if np.any(~np.isfinite(P[:,k])) else 0 for k in range(T.shape[1])]); Z=np.where(np.isfinite(Z),Z,avg)
  rr=th[2]+Z@th[3:]; A=np.c_[np.ones(len(rr)),rr]; H=U[:,:2]-hold[:,None]*th[:2]; p=np.linalg.pinv(A); k=p@H; E=H-A@k; print(refname,imp,'linear chamber drift offsets beta',k.ravel(),'loo rms',np.sqrt(np.mean((E/(1-np.einsum('ij,ji->i',A,p))[:,None])**2,0)))
  if refname=='trte' and imp=='train_deploy':
   for name,B in [('quadratic r',np.c_[A,rr**2]),('coupling hold',np.c_[A,hold,hold*rr]),('all features',np.c_[A[:,0],Z]),('last history',np.c_[A,past[:,-1]]),('branch z',np.c_[A,z0]),('lab offset',np.c_[A,np.eye(4)[labs,1:]])]:
    p=np.linalg.pinv(B); e=(H-B@(p@H))/(1-np.einsum('ij,ji->i',B,p))[:,None]; print(name,'loo rms',np.sqrt(np.mean(e**2,0)))
