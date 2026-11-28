import itertools,pickle,runpy,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; tr=s['train']['traj']; w=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['w'][:len(y)]; n=len(y); d=np.arange(n); ids=list(itertools.combinations(range(4),2)); p=np.column_stack([w[:,i]*w[:,j] for i,j in ids]); q=np.c_[np.ones(n),p]; a=np.c_[q,w,w*w]; ls=lambda X,z:np.linalg.lstsq(X,z,rcond=None)[0]
r=y-a@ls(a,y); print('pair coefficients',ids,ls(q,y)); print('residual by strongest pair',np.c_[p[:,-1],y,r][np.argsort(p[:,-1])][::4].round(4)); print('noise percentiles',np.quantile(abs(r),[.1,.5,.9,1])); print('trajectory pair coef',ls(q,tr).round(4));
folds=[d[52:],np.argsort(np.linalg.norm(w,axis=1))[-18:]]+[vi for seed in range(10) for vi in np.array_split(np.random.default_rng(seed).permutation(d),6)]; splits=[(np.setdiff1d(d,v),v) for v in folds]
def score(f, target=y):
 e=np.array([np.mean(abs(f(t,v,target)-target[v])) for t,v in splits]); return np.r_[e[:2],e[2:].mean(),e[2:].max()].round(6)
def lin(X):return lambda t,v,z:X[v]@ls(X[t],z[t])
for name,X in [('raw quad',a),('cross',q),('four pairs',q[:,[0,1,3,4,6]])]:print(name,score(lin(X)),score(lin(X),tr))
cols=[]
for i,j in itertools.combinations_with_replacement(range(6),2):
 for k,feat in [('quartic',p[:,i]*p[:,j]),('sixth',p[:,i]**2*p[:,j])]:
  X=np.c_[q,feat]; cols.append((score(lin(X)),(k,i,j)))
for e,k in sorted(cols,key=lambda item:item[0][2])[:12]:print(k,e)
def fit_rational(t,v,z,ix,mode):
 u=ix**2
 def design(c):
  den=1+c*u
  return q/den[:,None] if mode==0 else np.c_[q[:,:-1],q[:,-1]/den]
 def loss(c):
  X=design(c)[t];return np.mean((X@ls(X,z[t])-z[t])**2)
 c=minimize_scalar(loss,bounds=(0,3),method='bounded').x; X=design(c); return X[v]@ls(X[t],z[t])
for j,x in enumerate([*w.T,*p.T,q@ls(q,y)]):
 for m in range(2):
  if j==10: continue
  print('rational',j,m,score(lambda t,v,z:fit_rational(t,v,z,x,m)))
