import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; Y=tr['traj']; x=tr['x']; y=tr['y']; print('t',t,'traj sv',np.linalg.svd(Y,compute_uv=False)); np.set_printoptions(precision=4,suppress=True)
for pk in ['panel_a','panel_b','panel_c']:
 p=s[pk]; u=p['u']; Z=np.c_[np.ones(len(u)),u];
 for k,v in p.items():
  if k=='u':continue
  c=np.linalg.lstsq(Z,v,rcond=None)[0]; e=v-Z@c; print(pk,k,'affine',c,'res',np.sort(np.linalg.norm(e,axis=1))); keep=np.linalg.norm(e,axis=1)<3; print('refit',np.linalg.lstsq(Z[keep],v[keep],rcond=None)[0]); print('nonlinear residual correlations',np.corrcoef(np.c_[e,u**2,u**3,np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T].T)[:4,4:]);
def fit(r,YY=Y):
 q=-np.expm1(-t[:,None]*np.exp(r)); a=np.linalg.lstsq(q,YY.T,rcond=None)[0].T;return q,a,(YY-a@q.T)
for r in ([.4,1.8],[.2,1,3],[.12,.55,1.4,4]):
 f=least_squares(lambda z:fit(z)[2].ravel(),np.log(r),bounds=(-5,4)); q,a,e=fit(f.x); print('time rates',np.exp(f.x),'MAE',np.mean(abs(e)),'coeff std',a.std(0),'corr',np.corrcoef(np.c_[x,y,a].T)[5:,:5]);
print('affine x-> outputs',np.linalg.lstsq(np.c_[np.ones(len(x)),x],np.c_[y,Y],rcond=None)[0]); print('summary',tr['summary'],[y.mean(),y.std(),Y[:,-1].mean(),Y[:,-1].std()]);
