import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_a']['u']; v=s['panel_a']['v']; np.set_printoptions(precision=4,suppress=True)
def fit(X,y): return np.linalg.lstsq(X,y,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def trim(X,y,n=44):
 rng=np.random.default_rng(123); best=None
 for t in range(70):
  ix=rng.choice(len(y),X.shape[1]+1,replace=False) if t else np.arange(len(y))
  for _ in range(7):
   w=fit(X[ix],y[ix]); r=y-X@w; ix=np.argsort(abs(r))[:n]
  w=fit(X[ix],y[ix]); r=y-X@w; err=np.linalg.norm(r[ix]);
  if best is None or err<best[0]:best=(err,w,r,ix)
 return best
if __name__=='__main__':
 for j in range(4):
  a,b,c,d=np.roll(u,-j,axis=1).T; y=v[:,j]; pairs=[a*b,b*c,c*d,d*a,a*c,b*d]; base=aff(np.c_[a,b,c,d]);
  for name,X in [('aff',base),('one',np.c_[base,a*b]),('pair',np.c_[base,*pairs]),('full2',np.c_[base,*pairs,a*a,b*b,c*c,d*d]),('simple3',np.c_[base,a*b,a*a,b*b,a**3,b**3,a*a*b,a*b*b]),('simpleMix',np.c_[base,a*b,a*a,b*b,np.sin(2*a),np.sin(2*b),a*b*c,a*c*d,a*b*d,b*c*d])]:
   e,w,r,ix=trim(X,y); print(j,name,'rmse',e/np.sqrt(len(ix)),'coef',w,'r',np.sort(abs(r))[-8:]);
  e,w,r,ix=trim(np.c_[base,a*b],y); print('remaining correlations',np.corrcoef(np.c_[r[ix],(np.c_[a*a,b*b,c*c,d*d,a**3,b**3,a*b*c,a*c*d,a*b*d,b*c*d,a*a*b,a*b*b])[ix]].T)[0,1:]);
 print('range',u.min(0),u.max(0));print('summary actual',s['train']['y'].mean(),s['train']['y'].std(),s['train']['traj'][:,-1].mean(),s['train']['traj'][:,-1].std())
