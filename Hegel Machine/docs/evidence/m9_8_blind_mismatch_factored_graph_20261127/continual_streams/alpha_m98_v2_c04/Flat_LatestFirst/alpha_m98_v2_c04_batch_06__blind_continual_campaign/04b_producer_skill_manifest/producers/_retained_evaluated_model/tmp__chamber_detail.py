import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import alpha_m98_v2_c04_batch_04_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset']; tr=d['train']; te=d['test']; q=p['campaign_diagnostics']; t=d['eval_times']; dt=d['past_dt']; n=len(tr['lab_id']); b=m._fit_bridge(d['calibration_load_cells']); x=m._features(tr,te); tau,th=m._fit(x[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,hold=m._loads(q,b); Y=np.c_[q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']]; labs=q['lab_id']
X=m._features(tr,q)[n:]; Xs=m._features(q,q)[:len(Y)]; j={k:np.concatenate([tr[k],te[k],q[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}; Xj=m._features(j,j)[n+len(te['lab_id']):len(j['lab_id'])]; drifts=[th[2]+z@th[3:] for z in [X,Xs,Xj]]
later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; mem=lambda u:past@(-np.expm1(-dt[:,None]/u)*np.exp(-later[:,None]/u)); M=mem(tau); H=hold[:,None]; ones=np.ones(len(Y));
def tempor(u):return np.vstack([np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]])
A=tempor(np.array([.472,2.386])); row=[]
for y in Y: row.append(least_squares(lambda w:A@w-y,np.linalg.lstsq(A,y,rcond=None)[0],loss='soft_l1',f_scale=.01).x)
W=np.array(row); print('lab | hold | intrinsic memories | drifts (train+diag, diag only, all) | fitted fast minus theta0*hold | fitted slow | fitted drift');print(np.round(np.c_[labs,hold,M,*drifts,W[:,0]-th[0]*hold,W[:,1:]],3));print('missing rows',np.flatnonzero(np.isnan(q['coupon_features_raw']).any(axis=1)))
# Conditional linear model for amplitudes and drift; compare the entire law with and without memory and nonlinear drift terms.
def solve(select,fn,fast=False):
 def des(z):
  u=np.r_[tau[0],np.exp(z)] if fast else np.exp(z); a=tempor(u); cols,off=fn(u); return np.concatenate([a[None,:,i,None]*k[:,None,:] for i,k in enumerate(cols) if k.shape[1]],axis=2),off@a.T
 def fit(z,ret=False):
  D,off=des(z); a=D[select].reshape(-1,D.shape[-1]); y=(Y-off)[select].ravel(); w=np.linalg.lstsq(a,y,rcond=None)[0];
  for _ in range(7):
   v=(1+((a@w-y)/.014)**2)**(-.25); w=np.linalg.lstsq(a*v[:,None],y*v,rcond=None)[0]
  return (D@w+off,w) if ret else a@w-y
 z=np.log([2.4] if fast else [.472,2.4]); fitz=least_squares(fit,z,loss='soft_l1',f_scale=.014,max_nfev=70).x
 return *fit(fitz,True),np.exp(fitz)
def features(u,drift_type,struct):
 empty=np.empty((len(Y),0)); Z=drifts[drift_type]; dbase=np.c_[ones,Z]; MM=mem(u); off=np.zeros((len(Y),3));
 if struct=='full': return [np.c_[ones,H,past,dbase[:,1:]],np.c_[ones,H,past,dbase[:,1:]],dbase],off
 if struct=='relaxed': return [np.c_[ones,H,MM[:,0]],np.c_[ones,H,MM[:,1],Z],dbase],off
 if struct=='reset_nl': return [np.c_[ones,H],np.c_[ones,H,Z,np.minimum(Z,0),Z**2],np.c_[dbase,np.minimum(Z,0),Z**2]],off
 if struct=='reset': return [np.c_[ones,H],np.c_[ones,H,Z],dbase],off
 if struct=='memory_nl': return [np.c_[ones,H,MM[:,0]],np.c_[ones,H,MM[:,1],Z,np.minimum(Z,0),Z**2],np.c_[dbase,np.minimum(Z,0),Z**2]],off
 if struct=='memoryX': return [np.c_[ones,H,MM[:,0]],np.c_[ones,H,MM[:,1],(X,Xs,Xj)[drift_type]],np.c_[ones,(X,Xs,Xj)[drift_type]]],off
print('lab panel summaries (train+test, diagnostic-only)')
for l in np.unique(labs):
 for s in ([tr,te],[q]):
  rr=np.concatenate([z['ref_coupons'][z['lab_id']==l] for z in s]); print(l,len(rr),np.median(rr.mean(0)),np.median(rr.std(0)))
for fast in (True,False):
 for k in ('reset','relaxed','reset_nl'):
  fn=lambda u:features(u,1,k); mask=np.ones(len(Y),bool); yp,w,u=solve(mask,fn,fast=fast); err=yp-Y
  cv=np.empty_like(Y)
  for l in np.unique(labs): cv[labs==l]=(solve(labs!=l,fn,fast=fast)[0]-Y)[labs==l]
  print('fit',fast,k,u,np.round(w,4),'rms',np.sqrt(np.mean(err**2)), 'rate rows',np.round(err[:,-1],3)); print('LOLO, lab curve RMS/rate RMS',[(l,round(np.sqrt(np.mean(cv[labs==l,:-1]**2)),4),round(np.sqrt(np.mean(cv[labs==l,-1]**2)),4)) for l in np.unique(labs)])
for rw in (0,0.25,1):
 def model(p):
  tf,ts=np.exp(p[:2]); a=tempor(np.array([tf,ts])); R=drifts[1]; slow=p[3]+p[4]*hold+p[5]*R; drift=p[6]+p[7]*R; return np.c_[p[2]*hold,slow,drift]@a.T
 def res(p,sel):
  e=(model(p)-Y)[sel].copy(); e[:,-1]*=rw; return e.ravel()
 z=np.r_[np.log([.47,1.4]),th[0],-1,.43,1.08,.19,.14]; yp=least_squares(res,z,args=(slice(None),),loss='soft_l1',f_scale=.014,max_nfev=100).x
 e=model(yp)-Y
 print('strict fast hold-only',rw,'taus',np.exp(yp[:2]),'coeff',np.round(yp[2:],4),'time RMS',np.round(np.sqrt(np.mean(e**2,0)),4))
