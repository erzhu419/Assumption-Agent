import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from ros_probe import x,y,fit

def design(x,k,m):
 t=np.tanh(x*k)
 cols=[np.ones(len(x)),*x.T,*t.T]
 if 'q' in m: cols.extend(x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2))
 if 'd' in m: cols.extend((x*x).T)
 if 'p' in m: cols.extend(x[:,i]*x[:,j] for i,j in [(0,2),(1,3)])
 if 's' in m: cols.extend(np.sin(2*x).T)
 if 'f' in m: cols.extend((x/(1+x*x)).T)
 return np.column_stack(cols)

def solve(x,y,m):
 keep=fit(x,y,'affine')[1]; x,y=x[keep],y[keep]; w=np.ones(len(x)); k=np.full(4,1.5)
 for _ in range(5):
  def params(k):
   D=design(x,k,m); b=np.linalg.lstsq(D*w[:,None],y*w[:,None],rcond=None)[0]; return D,b
  def fun(k):
   D,b=params(k); return ((D@b-y)*w[:,None]).ravel()
  k=least_squares(fun,k,bounds=(.35,4),max_nfev=65).x; D,b=params(k); r=np.linalg.norm(D@b-y,axis=1); w=(1+(r/max(.004,1.5*np.median(r)))**2)**(-.25)
 return k,b

if __name__=='__main__':
 valid=fit(x,y,'affine')[1]; folds=np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5)
 for m in ('t','td','tp','tdp','tq','ts','tqs','tf','tqf'):
  k,b=solve(x,y,m); d=design(x,k,m); rr=y-d@b; errs=[]
  for ids in folds:
   tr=np.ones(len(x),bool); tr[ids]=False; k1,b1=solve(x[tr],y[tr],m); v=ids[valid[ids]]; errs.append(np.sqrt(np.mean((design(x[v],k1,m)@b1-y[v])**2)))
  print(m,'k',k.round(4),'RMS',np.sqrt(np.mean(rr[valid]**2,axis=0)).round(6),'folds',np.round(errs,6),flush=True)
  if m in ('t','tq'): print(b.round(5),flush=True)
