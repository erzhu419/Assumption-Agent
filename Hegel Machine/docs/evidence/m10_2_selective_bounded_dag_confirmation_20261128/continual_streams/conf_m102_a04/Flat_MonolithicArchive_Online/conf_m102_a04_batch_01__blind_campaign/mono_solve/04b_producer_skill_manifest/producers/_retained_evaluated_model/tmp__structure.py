import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True,linewidth=150)
def cv(F, reg=0, K=None):
 if K is None:
  if reg: p=F[52:]@np.linalg.solve(F[:52].T@F[:52]+reg*np.eye(F.shape[1]),F[:52].T@Y[:52]);
  else: p=F[52:]@fit(F[:52],Y[:52])
 else: p=K[52:,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),Y[:52]);
 return np.round(np.r_[np.mean(abs(p-Y[52:]),0),np.mean(abs((p[:,1:]-Y[52:,1:])@T.T))],5)
for name,z in [('x',x),('b',b),('v2',maps['v2'])]:
 for deg in (2,3,4,5):
  p=PolynomialFeatures(deg); F=p.fit_transform(z); C=fit(F[:52],Y[:52]); print(name,deg,cv(F), 'ridge', [(r,cv(F,r)) for r in [.001,.02,.1]])
  if deg==2: print(list(zip(p.get_feature_names_out(),C.round(4)))); print('train std',np.std(F[:52]@C-Y[:52],axis=0))
 D=cdist(z,z,'sqeuclidean');
 print('rbf',[(l,r,cv(None,r,np.exp(-D/l))) for l in [1,3,8,20] for r in [1e-5,.001]])
for key in ['v1','v2']:
 p=PolynomialFeatures(2); F=p.fit_transform(u); C=fit(F,c[key]); print('C poly',key,np.std(F@C-c[key]),list(zip(p.get_feature_names_out(),C.round(4))))
 print('C target',c[key][:12]); print('C input',u[:12]);
print('B',B)
