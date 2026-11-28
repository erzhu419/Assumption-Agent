import pickle,numpy as np,runpy,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; arch=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b03_batch_00_solver.py'); np.set_printoptions(precision=5,suppress=True,linewidth=145)
def ones(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
u,v=s['panel_a']['u'],s['panel_a']['v']; d=ones(u); mask=np.linalg.norm(v-d@ls(d,v),axis=1)<3; u,v=u[mask],v[mask]; print('clean',len(u),'range',np.min(u,axis=0),np.max(u,axis=0)); d=ones(u)
for degree in range(1,4):
 idx=[ix for deg in range(1,degree+1) for ix in itertools.combinations_with_replacement(range(4),deg)]; X=ones(np.array([np.prod(u[:,ix],axis=1) for ix in idx]).T); C=ls(X,v); I=np.eye(len(u))-X@np.linalg.pinv(X); err=I@v; print('deg',degree,'rms',np.mean(err**2,axis=0)**.5,'loo',np.mean((err/np.diag(I)[:,None])**2,axis=0)**.5); print('coeffs',[(ix,np.round(c,4).tolist()) for ix,c in zip([()]+idx,C) if max(abs(c))>.04]);
E=v-d@ls(d,v); cols={}
for i in range(4):
 for j in range(4):
  a,b=u[:,i],u[:,j]
  if i==j:
   for name,f in [('abs',abs),('square',np.square),('cube',lambda x:x**3),('tanh',np.tanh),('soft',lambda x:x/(1+abs(x))),('sin',np.sin),('cos',np.cos),('rational',lambda x:x/(1+x*x)),('sin2',lambda x:np.sin(2*x)),('cos2',lambda x:np.cos(2*x))]: cols[f'{name} {i}']=f(a)
  else:
   for name,f in [('prod',lambda a,b:a*b),('prod2',lambda a,b:a*b*b),('sinprod',lambda a,b:np.sin(a)*b),('tanhprod',lambda a,b:np.tanh(a)*b),('softprod',lambda a,b:a*b/(1+abs(b))),('absprod',lambda a,b:a*abs(b)),('ratio',lambda a,b:a/(1+b*b)),('sinsum',lambda a,b:np.sin(a+b)),('sindiff',lambda a,b:np.sin(a-b)),('cosdiff',lambda a,b:np.cos(a-b)),('cossum',lambda a,b:np.cos(a+b)),('sin_product',lambda a,b:np.sin(a*b))]: cols[f'{name} {i} {j}']=f(a,b)
F=np.stack(list(cols.values()),axis=1); G=F-d@ls(d,F); cor=(G.T@E)/(np.linalg.norm(G,axis=0)[:,None]*np.linalg.norm(E,axis=0));
for j in range(4):
 ix=np.argsort(-abs(cor[:,j]))[:14]; print(j,[(list(cols)[k],round(cor[k,j],4)) for k in ix]); print(np.c_[u,E[:,j]][:8])
