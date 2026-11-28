import pickle,numpy as np
from scipy.linalg import svd
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=[tr[k] for k in ('x','y','traj')]; t=s['eval_times']; print('times',t); np.set_printoptions(precision=4,suppress=True,linewidth=160)
def poly(x,d):
 return np.column_stack([np.prod(x[:,ii],axis=1) for k in range(d+1) for ii in combinations_with_replacement(range(x.shape[1]),k)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; a=poly(u,1)
 for key,v in p.items():
  if key=='u':continue
  mask=np.ones(len(u),bool)
  for i in range(8):
   w=fit(a[mask],v[mask]); errs=np.linalg.norm(a@w-v,axis=1); mask=errs<max(3*np.median(errs),.025)
  print(k,key,'coef\n',w,'rowerr',np.sort(errs)); print('quadratic resid',np.std(poly(u,2)@fit(poly(u,2)[mask],v[mask])-v,axis=0)); print('qcoef\n',fit(poly(u,2)[mask],v[mask])[5:])
  if k=='panel_a': A=w
U,D,V=svd(T,full_matrices=False); print('traj svd',D,'V',V[:4]); print('y explains traj or x');
for a in (T, x,T[:,[-1]],np.column_stack([x,T]),np.column_stack([x,T[:,[-1]]])):
 a=poly(a,1); print(np.std(a@fit(a,y)-y),fit(a,y))
z=poly(x,1)@A; yt=np.column_stack([y,T]);
for k in range(1,5):
 a=poly(z,k); w=fit(a[:56],yt[:56]); print('poly',k,'train',np.mean(abs(a[:56]@w-yt[:56]),axis=0),'test',np.mean(abs(a[56:]@w-yt[56:]),axis=0));
 if k<=2:print('coefficients y/first/last',w[:,[0,1,-1]])
