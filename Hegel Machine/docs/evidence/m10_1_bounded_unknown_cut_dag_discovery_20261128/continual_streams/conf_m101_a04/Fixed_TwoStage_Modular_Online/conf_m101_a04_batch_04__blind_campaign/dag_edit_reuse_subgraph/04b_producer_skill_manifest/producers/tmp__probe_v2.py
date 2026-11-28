import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; y=p['v2']; n,d=u.shape
np.set_printoptions(precision=4,suppress=True,linewidth=130)
a=np.c_[np.ones(n),u]; pairs=list(itertools.combinations(range(d),2)); cross=np.array([u[:,i]*u[:,j] for i,j in pairs]).T
q=np.c_[a,u*u,cross]; f1=np.linalg.lstsq(q,p['v1'],rcond=None)[0]; f2=np.linalg.lstsq(q,y,rcond=None)[0]; print('v1 coefficients\n',f1,'\nv2 quadratic coefficients\n',f2)
def metric(f,y=y):
 b=np.linalg.lstsq(f,y,rcond=None)[0]; e=y-f@b; h=(np.linalg.qr(f,mode='reduced')[0]**2).sum(1); return np.mean(abs(e)/np.maximum(1-h[:,None],1e-12)),np.mean(e*e)**.5
for name,fn in [('c',lambda t,x: np.cos(t*x)),('s',lambda t,x:np.sin(t*x)),('r',lambda t,x:1/(1+(t*x)**2)),('e',lambda t,x:np.exp(-(t*x)**2)),('cs',lambda t,x:np.c_[np.cos(t*x),np.sin(t*x)]),('ct',lambda t,x:np.c_[np.cos(t*x),np.tanh(t*x)]),('vel',lambda t,x:np.c_[np.cos(t*x),x*x])]:
 for bn,b in [('a',a),('cross',np.c_[a,cross]),('q',q)]:
  if b.shape[1]+fn(1,u).shape[1]>=n-1: continue
  grid=np.linspace(.3,8,78); fun=lambda t: metric(np.c_[b,fn(t,u)])[0]; ix=np.argmin([fun(t) for t in grid]); res=minimize_scalar(fun,bounds=(grid[max(0,ix-1)],grid[min(len(grid)-1,ix+1)]),method='bounded'); t=res.x
  print(name,bn,round(t,5),metric(np.c_[b,fn(t,u)]))
for i,j in pairs:
 for sign in (-1,1):
  w=u[:,i]+sign*u[:,j]
  for name,fn in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh)]:
   for t in (1,2,3):
    f=np.c_[a,np.cos(t*u),fn(t*w)]; loo,rmse=metric(f)
    if loo<.065: print('pair',i,j,sign,name,t,loo,rmse)
