exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.linalg import lstsq
from sklearn.linear_model import Lasso,Ridge,OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures
W=dict(np.load('/tmp/discover.npz')); np.set_printoptions(precision=4,suppress=True)
Y=np.c_[y,z]; I=np.arange(52); J=np.arange(52,72)
for n in ['q','b','c','d']:
 print(n,W[n].min(0),W[n].max(0));print('corr',np.corrcoef(np.c_[W[n],y,z[:,-1]].T)[-2:,:4])
print('y quant',np.quantile(y,[0,.1,.25,.5,.75,.9,1]),'resid extremes')
R=[]
for nm in ['qb','bc','qbc','qbd','bd','b','q']:
 A=np.column_stack([W[k] for k in nm]); names=[k+str(i) for k in nm for i in range(4)]
 for deg in [2,3]:
  P=PolynomialFeatures(deg,include_bias=False); F=P.fit_transform(A); labels=P.get_feature_names_out(names); S=np.maximum(F[I].std(0),.03); F=F/S
  for j in [0,8]:
   u=Y[:,j]
   for alg,pars in [(Lasso,[.0003,.001,.003,.01,.02,.03,.05,.08]),(Ridge,[.001,.01,.03,.07,.15,.3,.6,1.2,2.4,4.8]),(OrthogonalMatchingPursuit,[3,4,5,6,8,10,12,15,20,25])]:
    for p in pars:
     kw={'alpha':p} if alg!=OrthogonalMatchingPursuit else {'n_nonzero_coefs':p};
     if alg==Lasso:kw.update(max_iter=4000,tol=1e-5)
     fit=alg(**kw).fit(F[I],u[I]); pr=fit.predict(F); e=abs(pr-u); co=fit.coef_; top=np.argsort(abs(co))[-9:][::-1]; R.append((e[J].mean(),e[I].mean(),j,nm,deg,alg.__name__,p,[(labels[i],round(co[i]/S[i],3)) for i in top if co[i]!=0]));
for j in [0,8]:
 print('BEST',j); print(*sorted(r for r in R if r[2]==j)[:18],sep='\n')
