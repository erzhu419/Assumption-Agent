import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=150)
def ls(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def one(x):return np.c_[np.ones(len(x)),x]
print('times',s['eval_times'])
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; D=one(u)
 for k,v in s[p].items():
  if k=='u':continue
  keep=np.ones(len(u),bool)
  for _ in range(4):
   C=ls(D[keep],v[keep]); e=np.linalg.norm(v-D@C,axis=1); keep=e<max(.001,3*np.median(e))
  print(p,k,'coef',C,'row err',np.sort(e),'inliers',sum(keep))
  for f in [lambda v:v,lambda v:np.arctanh(np.clip(v,-.999999,.999999)),lambda v:np.sin(v),lambda v:np.sinh(v)]:
   t=f(v); X=PolynomialFeatures(2).fit_transform(u); print('poly2 RMS',np.sqrt(np.mean((t-X@ls(X,t))**2)))
x=s['train']['x']; t=s['train']['traj']; y=s['train']['y']; print('traj spectrum',np.linalg.svd(t,compute_uv=False))
for label,X in [('trajectory',one(t)),('trajectory+x',one(np.c_[t,x]))]+[(f'x poly {d}',PolynomialFeatures(d).fit_transform(x)) for d in range(1,5)]:
 C=ls(X[:52],np.c_[y,t][:52]); e=np.abs(X[52:]@C-np.c_[y,t][52:]); print(label,'mae',e.mean(0)); print('y weights',C[:,0] if len(C)<20 else '')
