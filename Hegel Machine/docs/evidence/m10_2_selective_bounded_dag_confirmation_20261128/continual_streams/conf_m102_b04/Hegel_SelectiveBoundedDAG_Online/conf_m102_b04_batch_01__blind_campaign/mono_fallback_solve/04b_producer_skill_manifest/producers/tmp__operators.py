import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement as C
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def basis(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],1) for k in range(1,d+1) for i in C(range(x.shape[1]),k)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def loo(x,y):
 e=y-x@fit(x,y); h=(np.linalg.svd(x,full_matrices=False)[0]**2).sum(1);return e/(1-h).reshape((-1,)+(1,)*(y.ndim-1))
def ident(v,a):return v
def signed(v,a):return v/(1-a*abs(v))
def isq(v,a):return v/np.sqrt(1-a*v*v)
def itan(v,a):return np.arctanh(np.sqrt(a)*v)/np.sqrt(a)
def isin(v,a):return np.arcsin(np.sqrt(a)*v)/np.sqrt(a)
def itan2(v,a):return np.tan(np.sqrt(a)*v)/np.sqrt(a)
def isinh(v,a):return np.sinh(np.sqrt(a)*v)/np.sqrt(a)
inv=[ident,signed,isq,itan,isin,itan2,isinh]
fwd={ident:ident,signed:lambda v,a:v/(1+a*abs(v)),isq:lambda v,a:v/np.sqrt(1+a*v*v),itan:lambda v,a:np.tanh(np.sqrt(a)*v)/np.sqrt(a),isin:lambda v,a:np.sin(np.sqrt(a)*v)/np.sqrt(a),itan2:lambda v,a:np.arctan(np.sqrt(a)*v)/np.sqrt(a),isinh:lambda v,a:np.arcsinh(np.sqrt(a)*v)/np.sqrt(a)}
def scan(D,v):
 R=[]
 for f in inv:
  mx=(.999/abs(v).max() if f==signed else .999/abs(v).max()**2) if f in (signed,isq,itan,isin) else (1.55/abs(v).max())**2 if f==itan2 else 15
  def score(a):
   b=f(v,a);return np.mean(loo(D,b)**2)/np.var(b)
  a=minimize_scalar(score,bounds=(1e-7,mx),method='bounded').x if f!=ident else 1.
  b=f(v,a);R.append((score(a)**.5,a,f,fit(D,b)))
 return sorted(R,key=lambda r:r[0])
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True,linewidth=150)
 p=s['panel_c'];u=p['u'];v=p['v1'];w=p['v2'];
 for name,D in [('u linear',basis(u,1)),('u quadratic',basis(u,2)),('v1 linear',basis(v,1)),('v1 quadratic',basis(v,2)),('u + v1 linear',np.c_[basis(u,1),v]),('u linear + v1 squares',np.c_[basis(u,1),v*v]),('u quad + v1 affine',np.c_[basis(u,2),v])]:
  for target,W in [('v1',v),('v2',w)]:
   if name.startswith('v1') and target=='v1' or ' + v1' in name and target=='v1':continue
   print(name,target,'raw LOO',np.std(loo(D,W),axis=0));
   for j in range(4):
    R=scan(D,W[:,j]);print(j,[(round(e,5),round(a,4),f.__name__) for e,a,f,b in R[:3]],'coeffs',R[0][-1])
 for name,D in [('u',u),('v1',v)]:
  print('quadratic v2',name,fit(basis(D,2),w).T)
