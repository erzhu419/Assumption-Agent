import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from sklearn.linear_model import LassoLars
from scipy.spatial.distance import cdist
p=s['panel_c']; u=p['u']; V=np.c_[p['v1'],p['v2']]; print('C sample',np.c_[u,V][:8]); print('C u norm',np.linalg.norm(u,axis=1));
def report(q,name):
 print(name,'ranges',q.min(0),q.max(0));
 for d in (2,3,4,5):
  F=PolynomialFeatures(d).fit_transform(q); K=F@F.T; L=np.linalg.lstsq(F[:n],Z[:n],rcond=None)[0]; print('poly',d,err(F[n:]@L));
  if d<=3:
   for j in range(3):
    v=F[:,1:]; sd=v[:n].std(0)+1e-9; model=LassoLars(alpha=.001).fit(v[:n]/sd,Z[:n,j]); print('lasso',j,round(np.mean(abs(model.predict(v[n:]/sd)-Z[n:,j])),5), sorted(zip(abs(model.coef_),np.array(PolynomialFeatures(d).fit(q).get_feature_names_out())[1:],model.coef_/sd),reverse=True)[:5]);
for a0 in ('ai','aib','aibi','x'):report(base[a0],a0)
for out in range(8):
 yy=V[:,out]; F=PolynomialFeatures(2).fit_transform(u); coef=ls(F,yy); H=np.zeros((4,4)); H[np.triu_indices(4)]=coef[5:]; H=(H+H.T)*.5; print('C quadratic',out,'error',np.std(F@coef-yy),'lin',coef[1:5],'eig',np.linalg.eigvalsh(H))
 for name,fun in [('sin',np.arcsin),('tanh',np.arctanh),('invquad',lambda v: v)]:
  results=[]
  for a in np.geomspace(max(abs(yy))*1.001, max(abs(yy))*100,80):
   z=fun(yy/a); g=aff(u); coef=ls(g,z); results.append((np.std(g@coef-z)/np.std(z),a,coef))
  print(name,min(results,key=lambda r:r[0]))
