import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar,least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def cols(u,degree): return PolynomialFeatures(degree).fit_transform(u)
for p in 'abc':
 d=s['panel_'+p]; u=d['u']; print('panel',p,'u range',u.min(0),u.max(0))
 for key,v in d.items():
  if key=='u': continue
  print(key,'range',v.min(0),v.max(0)); print(np.c_[u,v][:8]); print('robust affine and quadratic')
  for deg in (1,2):
   f=cols(u,deg); co=[]
   for j in range(4):
    n=len(u)-(6 if p=='a' else 0); y=v[:,j]; z=ls(f,y)
    for i in range(10):
     q=np.argsort(abs(f@z-y))[:n]; z=ls(f[q],y[q]);
    co.append(z); print(j,'med,trim-rms,max',np.median(abs(f@z-y)),np.sqrt(np.mean((f[q]@z-y[q])**2)),np.max(abs(f@z-y)))
   print(np.array(co).T)
  print('LOO errors:')
  for name,f in [('aff',aff(u)),('quadratic',cols(u,2)),('diagquad',np.c_[aff(u),u*u]),('diagcube',np.c_[aff(u),u*u,u**3]),('diagtrig',np.c_[aff(u),np.sin(u),np.cos(u)])]:
   y=v; h=np.diag(f@np.linalg.pinv(f)); err=(f@ls(f,y)-y)/(1-h[:,None]); print(name,np.mean(abs(err),axis=0))
print('TIME',s['eval_times']); t=s['train']['traj']; y=s['train']['y']; x=s['train']['x']; print('y corr time',np.corrcoef(np.c_[y,t].T)[0]); print('affine train fit for x to y and traj',ls(aff(x),np.c_[y,t])); print('SVD time loadings',np.linalg.svd(t,full_matrices=False)[2][:4]); print('test/training correlation',np.corrcoef(x.T))
