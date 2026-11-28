import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; T=tr['traj']; x=tr['x']; y=tr['y']; np.set_printoptions(precision=4,suppress=True)
def aff(u,v):
 h=np.c_[np.ones(len(u)),u]; b=np.linalg.lstsq(h,v,rcond=None)[0]; return b,v-h@b
for p in ['panel_a','panel_b','panel_c']:
 d=s[p]; u=d['u'];
 for k,v in d.items():
  if k=='u': continue
  b,r=aff(u,v); print(p,k,b, 'row error',np.sort(np.linalg.norm(r,axis=1)),sep='\n')
  if p=='panel_a':
   keep=np.linalg.norm(r,axis=1)<2; b,r=aff(u[keep],v[keep]); print('robust',b,np.std(r,axis=0))
print('time svd',np.linalg.svd(T,compute_uv=False))
def fit(r,t=t,T=T):
 q=-np.expm1(-t[:,None]*np.exp(r)); a=np.linalg.lstsq(q,T.T,rcond=None)[0]; return a,(T.T-q@a)
for n in [1,2,3]:
 o=least_squares(lambda p:fit(p)[1].ravel(),np.linspace(-1,1,n)); a,e=fit(o.x); print('modes',np.exp(o.x),np.std(e), 'coef std',a.std(1));
 print('corr x/y/modes',np.corrcoef(np.c_[x,y,a.T].T)[5:]);
