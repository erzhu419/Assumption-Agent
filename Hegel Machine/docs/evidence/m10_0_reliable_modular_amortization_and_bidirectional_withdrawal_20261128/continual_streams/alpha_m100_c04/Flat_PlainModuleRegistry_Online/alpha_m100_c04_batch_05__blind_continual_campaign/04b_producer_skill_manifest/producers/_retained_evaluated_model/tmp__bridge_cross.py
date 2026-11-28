import pickle,numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; s=c['single_channel_sweeps']; x=c['true_channel_loads']; y=c['bridge_raw']; g=np.max(abs(y),axis=1)<5; x,y=x[g],y[g]; axis=(x!=0).sum(1)==1
families={'cos':lambda u,k:np.cos(k*u)-1,'rational':lambda u,k:u*u/(1+(k*u)**2),'exp':lambda u,k:np.expm1(-(k*u)**2),'tanh^2':lambda u,k:np.tanh(k*u)**2,'sin':lambda u,k:np.sin(k*u),'tanh':lambda u,k:np.tanh(k*u)}
for j in range(4):
 i=(j+2)%4; m=(s['lab_id']==j)&(abs(s['cmd_stress'])<5); t=s['true_stress'][m]; z=s['cmd_stress'][m]; xx=np.vstack((x,np.eye(4)[j]*t[:,None])); yy=np.r_[y[:,j],z]; n=len(x); rng=np.random.default_rng(20); folds=np.array_split(rng.permutation(n),8)
 print('channel',j,'coupled to',i,flush=True)
 for name,f in families.items():
  for direction in ('additive','ridge','difference_only'):
   def fit(mask):
    def profile(p):
     own=xx[:,j]; u=xx[:,i]; a=np.column_stack([np.ones(len(xx)),xx,np.sin(p[0]*own), f(u+p[2]*own,p[1])-(f(p[2]*own,p[1]) if direction=='ridge' else 0)])
     b=np.linalg.lstsq(a[mask],yy[mask],rcond=None)[0]; return a@b-yy,b
    bounds=([2.2,.12,-2],[2.8,6,2]); p=min([least_squares(lambda p:profile(p)[0][mask], [2.5,k,d], bounds=bounds,max_nfev=100) for k in (.55,1.25,2.1,3.2) for d in ([0] if direction=='additive' else [-.5,.5])],key=lambda a:a.cost).x if direction!='additive' else np.r_[min([least_squares(lambda p:profile(np.r_[p,0])[0][mask],[2.5,k],bounds=([2.2,.12],[2.8,6]),max_nfev=100) for k in (.55,1.25,2.1,3.2)],key=lambda a:a.cost).x,0]
    return *profile(p),p
   err,b,p=fit(np.ones(len(xx),bool));
   if np.std(err)>.012: continue
   cv=np.zeros(n)
   for fold in folds:
    mask=np.ones(len(xx),bool); mask[fold]=False; cv[fold]=fit(mask)[0][fold]
   print(name,direction,'fit/CV max axis/mixed',np.round([np.std(err),np.max(abs(cv[axis])),np.max(abs(cv[~axis]))],7),'p',np.round(p,5),'b',np.round(b,5),flush=True)
