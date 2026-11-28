import pickle, numpy as np
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=120)
def fit(a,b):
 p=np.linalg.lstsq(a,b,rcond=None)[0]; r=b-a@p; h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1); e=r/np.maximum(1-h,1e-10).reshape((-1,)+(1,)*(r.ndim-1))
 return p, np.mean(abs(r),axis=0), np.mean(abs(e),axis=0), np.max(abs(e),axis=0)
for panel in ('panel_a','panel_b','panel_c'):
 u=s[panel]['u']
 for k,v in s[panel].items():
  if k=='u': continue
  print(panel,k, 'ranges', np.min(u,axis=0), np.max(u,axis=0), np.min(v,axis=0), np.max(v,axis=0))
  for label,x,y in [('forward',u,v),('inverse',v,u)]:
   for deg in [1,2,3,5,7]:
    e=[fit(np.vander(a,deg+1,increasing=True),b) for a,b in zip(x.T,y.T)]; print(label,deg,'diag LOO MAE',[round(a[2],5) for a in e], 'LOO max', [round(a[3],5) for a in e]);
    if label=='forward' and deg in (1,3,5): print('coeff',np.stack([a[0] for a in e]).T)
   for deg in (1,2,3):
    xx=PolynomialFeatures(deg).fit_transform(x)
    if xx.shape[1]>=len(x): continue
    p,r,e,m=fit(xx,y); print(label,'full',deg,'resid',r,'LOO',e,'max',m)
    if deg==1 and label=='forward': print(p)
print('train and test x ranges',[(s[k]['x'].min(axis=0),s[k]['x'].max(axis=0)) for k in ['train','test']]); print('traj spectrum',np.linalg.svd(s['train']['traj'],compute_uv=False))
