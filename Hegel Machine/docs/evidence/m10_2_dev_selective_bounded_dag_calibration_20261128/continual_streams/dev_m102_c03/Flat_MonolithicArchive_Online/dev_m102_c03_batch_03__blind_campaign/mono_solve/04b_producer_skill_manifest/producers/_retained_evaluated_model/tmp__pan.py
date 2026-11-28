import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def feat(u):
 return np.c_[np.ones(len(u)),u,u**2,*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)],u**3,u**4]
for p in ['panel_a','panel_c']:
 u=s[p]['u'];
 for k,v in s[p].items():
  if k=='u':continue
  h=feat(u); keep=np.ones(len(u),bool)
  if p=='panel_a':
   r=v-h[:,:5]@np.linalg.lstsq(h[:,:5],v,rcond=None)[0]; keep=np.linalg.norm(r,axis=1)<3
  for m in [5,9,15,19,23]:
   a=h[keep,:m]; b=np.linalg.lstsq(a,v[keep],rcond=None)[0]; r=v[keep]-a@b; loo=r/(1-np.sum(a*np.linalg.pinv(a).T,axis=1,keepdims=True)); print(p,k,m,'rmse',np.sqrt(np.mean(r**2)), 'loo',np.sqrt(np.mean(loo**2))); 
   if m==15: print(b)
