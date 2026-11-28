exec(open('/tmp/structure.py').read().split('Y=[]; T=[]')[0])
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit, Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
q=lambda zz: np.c_[np.ones(len(zz)),zz[:,0]*zz[:,1],zz[:,0]*zz[:,3],zz[:,1]*zz[:,2],zz[:,2]*zz[:,3]]
qc=ls(q(u),p['v1']); c=q(z)@qc; d=m._branch(p,z,2); bz=ones(z)@B
print('SHAPES',[(k,[(j,np.shape(v)) for j,v in a.items()] if isinstance(a,dict) else a) for k,a in s.items()]); print('SCORES', 'z range', np.min(z),np.max(z))
ts=s['eval_times']; h=lambda rates:-np.expm1(-ts[:,None]*rates); rates=least_squares(lambda r:(h(r)@ls(h(r),t.T)-t.T).ravel(),[1.8,.4],bounds=([1,.1],[3,.95])).x
print('rates',rates, 'projection errors',np.mean(abs(h(rates)@ls(h(rates),t.T)-t.T)), 'times',ts)
amp=ls(h(rates),t.T).T
suite={}
for k,zz in [('z',z),('B',bz),('Binv',coords['Binv'])]:
 for deg in [1,2,3,4,5]:
  poly=PolynomialFeatures(deg,include_bias=False).fit(zz);suite[k+str(deg)]=(poly.transform(zz),poly.get_feature_names_out([k+str(i) for i in range(4)]))
 pf=PolynomialFeatures(2,include_bias=False).fit(np.c_[zz,c,d]); suite[k+'cd2']=(pf.transform(np.c_[zz,c,d]),pf.get_feature_names_out([f'{p}{i}' for p in (k,'c','d') for i in range(4)]))
for k in ['Ainv','B','Binv']:
 zz=coords[k];cc=q(zz)@qc;dd=m._branch(p,zz,2)
 pf=PolynomialFeatures(3,include_bias=False).fit(np.c_[cc,dd]);suite[k+'cd3']=(pf.transform(np.c_[cc,dd]),pf.get_feature_names_out([f'{p}{i}' for p in ('c','d') for i in range(4)]))
for target in [y,amp[:,0],amp[:,1]]:
 results=[]
 for k,(f,names) in suite.items():
  for typ,lam in [('ols',0)]+[('ridge',l) for l in [.001,.03,.3,3]]+[('lasso',l) for l in [.001,.005,.02,.06]]+[('omp',l) for l in [3,5,8,12,18,24] if l<=f.shape[1]]:
   vals=[]; subsets=[]
   for r,v in cuts:
    mu=f[r].mean(0); sig=f[r].std(0)+1e-12; R=(f[r]-mu)/sig; V=(f[v]-mu)/sig; yy=target[r]; ym=yy.mean();
    if typ=='ols': cf=ls(R,yy-ym)
    if typ=='ridge': cf=Ridge(alpha=lam,fit_intercept=False).fit(R,yy-ym).coef_
    if typ=='lasso': cf=Lasso(alpha=lam,fit_intercept=False,max_iter=5000,tol=5e-5).fit(R,yy-ym).coef_
    if typ=='omp': cf=OrthogonalMatchingPursuit(n_nonzero_coefs=lam,fit_intercept=False).fit(R,yy-ym).coef_
    vals.append(np.mean(abs(V@cf+ym-target[v]))); subsets.append(names[np.argsort(abs(cf))[-8:][::-1]])
   results.append((max(vals),np.mean(vals),k,typ,lam,np.round(vals,4),subsets[0]))
 print('TOP TARGET');
 for entry in sorted(results,key=lambda q:q[:2])[:12]: print(entry)
