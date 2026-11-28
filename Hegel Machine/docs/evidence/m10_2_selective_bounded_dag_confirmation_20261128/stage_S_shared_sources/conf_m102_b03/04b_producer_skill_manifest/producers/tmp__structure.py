from gp import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd
U,S,V=svd(T[tr],full_matrices=False); Y=np.c_[y,T@V[:2].T]
for names in [('xaC',),('xaD',),('xaC','xaD'),('xaC','xaD','xaB')]:
 for degrees in itertools.product([1,2,3],repeat=len(names)):
  D=[]
  for k,p in zip(names,degrees):
   D.append(PolynomialFeatures(p,include_bias=False).fit_transform(z[k]))
  D=np.column_stack(D); D=(D-D[tr].mean(0))/D[tr].std(0)
  A=np.c_[np.ones(len(D)),D]; uu,ss,vv=svd(A[tr],full_matrices=False)
  row=[]
  for lam in [1e-8,1e-5,.001,.03]:
   shrink=ss**2/(ss**2+lam); co=vv.T@((shrink/ss)[:,None]*(uu.T@Y[tr])); e=A@co-Y
   loo=np.mean(abs(e[tr]/np.maximum(1-(uu**2)@shrink,1e-9)[:,None]),0); mae=np.mean(abs(e[va]),0); maeT=np.mean(abs(e[va,1:]@V[:2]),0).max()
   row.append((lam,loo,mae,round(maeT,5),np.std(e[tr],0)))
  if min(r[2][0] for r in row)<.049 or min(r[3] for r in row)<.036:print(names,degrees,row)
