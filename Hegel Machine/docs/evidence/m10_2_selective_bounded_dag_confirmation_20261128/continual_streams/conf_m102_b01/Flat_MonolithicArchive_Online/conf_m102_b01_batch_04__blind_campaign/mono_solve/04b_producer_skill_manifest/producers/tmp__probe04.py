import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
print('times',s['eval_times']);print('svd',np.linalg.svd(np.c_[s['train']['y'],s['train']['traj']],compute_uv=False))
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; x=p['u'];
 for v,y in p.items():
  if v=='u':continue
  print(k,v)
  for deg in [1,2]:
   z=PolynomialFeatures(deg).fit_transform(x);q=ls(z,y);e=z@q-y;h=np.sum(z*np.linalg.pinv(z).T,axis=1);print(deg,'RMSE',np.sqrt(np.mean(e**2,axis=0)),'LOO',np.sqrt(np.mean((e/(1-h[:,None]))**2,axis=0)));print(q if deg==1 else '')
  if k=='panel_a':
   for j in range(4):
    z=aff(x[:,j:j+1]);q=ls(z,y[:,j]);
    for _ in range(15):
     i=np.argsort(np.abs(z@q-y[:,j]))[:39];q=ls(z[i],y[i,j]);
    print('robust',j,q,np.sort(np.abs(z@q-y[:,j]))[::3]);
