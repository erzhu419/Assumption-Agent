import pickle
import numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
T=D['eval_times']; DT=D['past_dt']; TR=D['train']; P=D['chamber_verification_trials']; TE=D['test']
c=D['calibration_load_cells']; raw=c['bridge_raw']; true=c['true_channel_loads']; inv=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],true,rcond=None)[0].sum(axis=1)
def split(s,mask):return {k:v[mask] for k,v in s.items()}
def features(s,refs,fill_rows):
 def align(a):
  x=a['coupon_features_raw'].copy()
  for lab in np.unique(a['lab_id']):
   panel=np.concatenate([r['ref_coupons'][r['lab_id']==lab] for r in refs]); sel=a['lab_id']==lab
   x[sel]=(x[sel]-np.median(panel.mean(0)))/max(1e-6,np.median(panel.std(0)))
  return x
 x=align(s); z=align(fill_rows); _,cell,cnt=np.unique(np.c_[fill_rows['lab_id'],fill_rows['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/cnt[cell,None]; fill=np.nansum(z*w,axis=0)/np.sum(np.isfinite(z)*w,axis=0)
 return np.c_[np.ones(len(x)),np.where(np.isfinite(x),x,fill)]
def basis(s,x,taus,mode='kelvin',extra=None):
 past=s['past_channel_raw']@inv[:4]+inv[-1]; hold=s['hold_channel_raw']@inv[:4]+inv[-1]; taus=np.array(taus)
 ds=DT[:,None]/taus; decay=np.exp(-ds); amt=-np.expm1(-ds); state=(past[:,:,None]*amt*np.vstack([np.exp(-np.cumsum(ds[:0:-1],axis=0)[::-1]),np.ones((1,len(taus)))])).sum(1)
 if mode=='reverse':state=(past[:,::-1,None]*amt*np.vstack([np.exp(-np.cumsum(ds[:0:-1],axis=0)[::-1]),np.ones((1,len(taus)))])).sum(1)
 amp=hold[:,None]-state if mode not in ('hold','memory','plus') else {'hold':hold[:,None]+0*state,'memory':state,'plus':hold[:,None]+state}[mode]
 if extra=='hold':x=np.c_[x,hold]
 if extra=='tele':x=np.c_[x,s['aux_chamber_telemetry']]
 if extra=='lab':x=np.c_[x,(s['lab_id'][:,None]==[1,2,3]).astype(float)]
 A=np.concatenate([amp[:,None,:]*(-np.expm1(-T[None,:,None]/taus)),x[:,None,:]*T[None,:,None]],axis=2)
 R=np.c_[amp/taus,x]; return A,R
def fit(s,x,mode='kelvin',extra=None,scales=(.01,.015),starts=((.35,2.8),(.1,7),(1.3,12))):
 y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; sy,sr=scales; best=None
 def residual(v):
  A,R=basis(s,x,v[:2],mode,extra); return np.r_[((A@v[2:]-y)/sy).ravel(),(R@v[2:]-r)/sr]
 for tau in starts:
  A,R=basis(s,x,tau,mode,extra); z=np.linalg.lstsq(np.vstack([A.reshape(-1,A.shape[-1])/sy,R/sr]),np.r_[y.ravel()/sy,r/sr],rcond=None)[0]; n=len(z)
  f=least_squares(residual,np.r_[tau,z],bounds=(np.r_[[.050001]*2,[-np.inf]*n],np.r_[[24.99999]*2,[np.inf]*n]),loss='cauchy',max_nfev=160,x_scale='jac')
  if best is None or f.cost<best.cost:best=f
 v=best.x.copy(); order=np.argsort(v[:2]);v[:2]=v[:2][order];v[2:4]=v[2:4][order];return v,best.cost

def score(s,x,v,mode='kelvin',extra=None):
 A,R=basis(s,x,v[:2],mode,extra); dy=A@v[2:]+s['strain_0'][:,None]-s['strain_curves_obs']; dr=R@v[2:]-s['instant_rate_obs']; return [np.median(abs(dy)),np.sqrt(np.mean(dy**2)),np.median(abs(dr)),np.sqrt(np.mean(dr**2)),np.median(dy),np.median(dr)]
if __name__=='__main__':
 for s in (TR,P):
  print('SAMPLE',len(s['lab_id']))
  for name,refs in [('pool',[TR,P,TE]),('train',[TR]),('self',[s])]:
   for mode,extra in [('kelvin',None),('kelvin','hold'),('kelvin','lab'),('kelvin','tele'),('reverse',None),('hold',None),('plus',None),('memory',None)]:
    if extra=='tele' and 'aux_chamber_telemetry' not in s:continue
    if name!='pool' and (mode,extra)!=('kelvin',None):continue
    x=features(s,refs,TR); v,cost=fit(s,x,mode,extra)
    print(name,mode,extra,round(cost,2),np.round(v,5),np.round(score(s,x,v,mode,extra),5),flush=True)
  x=features(s,[TR,P,TE],TR); v,_=fit(s,x); other=P if s is TR else TR; print('TRANSFER',np.round(score(other,features(other,[TR,P,TE],TR),v),5))
