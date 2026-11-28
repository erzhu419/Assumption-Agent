import pickle,sys
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a01_batch_03_solver as old
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; A=old._affine
z,h=old._coordinates(s,x); b=np.linalg.lstsq(A(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]; print('b singular values',np.linalg.svd(b[1:],compute_uv=False))
c=s['panel_c']; coords={'x':x,'z':z,'h':h,'Bz':A(z)@b};
for key in ('v1','v2'):
 def f(w):return w/(1+.84*abs(w)) if key=='v1' else np.sqrt(1+.959**2*w*w)
 def basis(w): return np.c_[A(w),f(w)]
 cf=np.linalg.lstsq(basis(c['u']),c[key],rcond=None)[0];print(key,'residual',np.std(c[key]-basis(c['u'])@cf,axis=0),'coef',cf.round(4))
 for name,w in list(coords.items())[:4]: coords[name+key[-1]]=basis(w)@cf
print('coord ranges',[(k,np.round(w.min(0),2),np.round(w.max(0),2)) for k,w in coords.items()]);
Y=np.c_[y,t]; rng=np.random.default_rng(123); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(9)]
def evaluate(K):
 a=[]
 for j in splits:
  i,v=j[:52],j[52:]; p=K[np.ix_(v,i)]@cho_solve(cho_factor(K[np.ix_(i,i)]),Y[i]);e=np.abs(p-Y[v]); a.append([e[:,0].mean(),e[:,1:].mean()]);
 a=np.array(a);return np.r_[a[0],a.mean(0),a.max(0)]
for k,w in coords.items():
 w=w/np.std(w,axis=0); G=w@w.T/4; D=cdist(w,w,'sqeuclidean'); base=1+G
 models={}
 for d in range(2,7):
  for rho in (.08,.25,.7,1.5):
   for noise in (1e-5,1e-3,.01): models[('poly',d,rho,noise)]=evaluate(sum((rho**(j-1) if j>1 else 1)*G**j for j in range(d+1))+noise*np.eye(len(w)))
 for scale in (.65,1,1.5,2.2,3.5,5):
  for noise in (1e-5,1e-3,.01):models[('rbf',scale,noise)]=evaluate(base+.15*np.exp(-.5*D/scale**2)+noise*np.eye(len(w)))
 for i in (2,3):
  print(k,'yt'[i-2],[(j, np.round(m,4)) for j,m in sorted(models.items(),key=lambda jm:jm[1][i])[:2]])
