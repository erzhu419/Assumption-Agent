from bridge_probe import X,Y,N,basis
from itertools import product
import numpy as np
from scipy.optimize import minimize

# The three gross bridge spikes are separated from all other measurements by an order of magnitude.
clean=np.max(np.abs(Y-np.median(Y,axis=0)),axis=1)<8
pair=[2,3,0,1]
forms={'tanh':np.tanh,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'root':lambda z:z/np.sqrt(1+z*z),'sin':np.sin,'erf':__import__('scipy').special.erf}

def fit(m,j,diag,cross=None,degree=2,shift=False):
 def des(x,p):
  u=x[:,j]; v=x[:,pair[j]]
  cols=[basis(x,'quadratic' if degree==2 else 'affine'),forms[diag](p[0]*u+(p[-1] if shift else 0.))]
  if cross=='cubic': cols.append(v**3)
  elif cross: cols.append(forms[cross](p[1]*v))
  return np.column_stack(cols)
 def loss(p):
  a=des(X[m],p); b=np.linalg.lstsq(a,Y[m,j],rcond=None)[0];return np.sum((a@b-Y[m,j])**2)
 bounds=[(.15,5)]*(1+int(cross in forms))+([(-1,1)] if shift else [])
 p=min((minimize(loss,[start]*(len(bounds)-int(shift))+([0] if shift else []),bounds=bounds,method='L-BFGS-B',options={'ftol':1e-13,'gtol':1e-10}) for start in (.65,1.6)),key=lambda r:r.fun).x
 b=np.linalg.lstsq(des(X[m],p),Y[m,j],rcond=None)[0]
 return des(X,p)@b,p,b

if __name__=='__main__':
 folds=np.random.default_rng(8).permutation(N)%5
 edge=np.max(abs(X),axis=1)>=np.quantile(np.max(abs(X),axis=1),.8)
 rows=[]
 for diag,cross,degree,shift in [(d,c,2,s) for d,c,s in product(forms,(None,'cubic','tanh','atan','rat'),(False,))]+[(d,c,2,True) for d,c in product(('tanh','atan','rat'),(None,'tanh'))]:
  res=[fit(clean,j,diag,cross,degree,shift) for j in range(4)]
  rm=np.sqrt(np.mean((np.column_stack([r[0] for r in res])[clean]-Y[clean])**2))
  rows.append((rm,diag,cross,degree,shift,res))
 for rm,d,c,g,s,res in sorted(rows,key=lambda row:row[0])[:12]:
  pred=np.zeros_like(Y)
  for f in range(5):
   for j in range(4): pred[folds==f,j]=fit(clean&(folds!=f),j,d,c,g,s)[0][folds==f]
  ep=np.column_stack([fit(clean&~edge,j,d,c,g,s)[0] for j in range(4)])
  print(d,c,s,'rms',round(rm,7),'OOF max',round(np.max(np.linalg.norm((pred-Y)[clean],axis=1)),6),'edge max',round(np.max(np.linalg.norm((ep-Y)[clean&edge],axis=1)),6),'p',np.round([r[1] for r in res],4).tolist())
 rm,d,c,g,s,res=min(rows,key=lambda row:row[0]);print('best coeffs:\n',np.round(np.column_stack([r[2] for r in res]),6)); print('best residual quantiles',np.quantile(abs((np.column_stack([r[0] for r in res])-Y)[clean]),[.5,.9,1],axis=0))
