exec(open("/tmp/chamber_probe.py").read().split("for k in (1,2,3,4):")[0])
from itertools import combinations
z=old._memory(taus,past,dt); gaps=hold[:,None]-z
print('curve/rate shape',Y.shape,'train/test load ranges',[(np.min(s['hold_channel_raw'].sum(-1)),np.max(s['hold_channel_raw'].sum(-1))) for s in (a,b)])
for j in range(2):
 print('mode fast ratio',j,np.percentile(profile(np.log([.50633129,3.32304303]),True)[0][:,0]/[hold,gaps[:,0]][j],[0,25,50,75,100]))
print('diagnostic raw feature NaNs',np.isnan(c['coupon_features_raw']).sum(axis=0))
A=np.c_[np.ones(len(Y)),hold,drift]; wt=np.r_[np.ones(len(t)),1.]
labels=['affine','affine+memory','affine+past','affine+features','quadratic','drift+gaps','hold+drift']
models=[A,np.c_[A,z],np.c_[A,past],np.c_[A,x[n+m:]],np.c_[A,hold**2,hold*drift,drift**2],np.c_[np.ones(len(Y)),drift,gaps],A[:,1:]]

def fit(X,ix,mode='free'):
 def calc(v):
  T=np.exp(v) if mode=='free' else (np.r_[taus[0],np.exp(v)] if mode=='fast_fixed' else taus)
  B=basis(T); W=np.linalg.lstsq(B*wt[:,None],np.linalg.lstsq(X[ix],Y[ix]*wt,rcond=None)[0].T,rcond=None)[0].T
  return (X@W@B.T-Y)*wt,W,T
 v0=np.log([.51,3.3] if mode=='free' else [3.3]); v=v0 if mode=='both_fixed' else least_squares(lambda v:calc(v)[0][ix].ravel(),v0,bounds=(np.log(.06),np.log(20)),max_nfev=100).x
 return calc(v)
for label,X in zip(labels,models):
 for mode in ('free','fast_fixed','both_fixed'):
  ix=np.arange(len(Y)); R,W,T=fit(X,ix,mode); oof=np.empty_like(R)
  for lab in np.unique(c['lab_id']):
   v=c['lab_id']==lab; oof[v]=fit(X,ix[~v],mode)[0][v]
  print(label,mode,'T',T.round(6),'fit',np.sqrt((R**2).mean()).round(6),'cv',np.sqrt((oof**2).mean(axis=0))[[0,len(t)-1,len(t)]].round(6),'worst',max(np.sqrt(np.mean(oof[c['lab_id']==lab]**2)) for lab in np.unique(c['lab_id'])).round(6))
  if label in ('affine','hold+drift') and mode in ('free','both_fixed'): print('coef',W.round(6))
print('prior batch 01 physical prediction RMSE:')
import dev_alpha_m99_00_batch_01_solver as m1
for mode in ('archived_scale','current_projection'):
 T=taus*m1._CHAMBER_TIME_SCALE
 R=A@m1._CHAMBER_COEFFICIENTS.T@basis(T).T-Y;print(mode,T,np.sqrt((R**2).mean(0))[[0,-2,-1]])
 break
