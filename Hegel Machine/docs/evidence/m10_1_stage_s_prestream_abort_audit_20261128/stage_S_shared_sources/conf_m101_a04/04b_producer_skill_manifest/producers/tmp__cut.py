import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/links.py').read())
from sklearn.linear_model import Ridge,OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import erf
H,J=[app(z,op) for op in ops2]; W=np.c_[H,J]; print('panel1 row space',np.linalg.svd(ops2[0][-1][5:])[1]); print('z bounds',z.min(0),z.max(0));
opts={}
for nm,L in [('HJ',W),('zHJ',np.c_[z,W]),('intrinsic',np.c_[np.arcsinh(z*ops2[0][2][:4]+ops2[0][2][4:]),np.logaddexp(0,z*2.2)]),('basic',np.c_[z,np.sin(1.5*z),np.tanh(2*z),np.logaddexp(0,2.2*z)])]:
 for deg in (1,2,3):
  polyf=PolynomialFeatures(deg,include_bias=False); E=polyf.fit_transform(L); labels=polyf.get_feature_names_out(); E=(E-E.mean(0))/np.maximum(E.std(0),1e-10)
  for alpha in (.0001,.001,.01,.1,1):
   pp=np.empty_like(Y)
   for a,c in folds: pp[c]=Ridge(alpha=alpha).fit(E[a],Y[a]).predict(E[c]);
   er=np.c_[abs(pp[:,2]-y),abs(pp[:,:2]@D.T-T)];opts[nm,deg,alpha]=(er[:,0].mean(),er[:,1:].mean(0).max())
  if deg==2:
   for n in (3,6,10,15,22):
    pp=np.empty_like(Y)
    for a,c in folds:
     for j in range(3): pp[c,j]=OrthogonalMatchingPursuit(n_nonzero_coefs=n).fit(E[a],Y[a,j]).predict(E[c]);
    er=np.c_[abs(pp[:,2]-y),abs(pp[:,:2]@D.T-T)];opts[nm,deg,'omp'+str(n)]=(er[:,0].mean(),er[:,1:].mean(0).max())
   for j in range(3):
    a=Ridge(alpha=.01).fit(E,Y[:,j]); ii=np.argsort(abs(a.coef_))[-8:][::-1]; print(nm,j,list(zip(labels[ii],a.coef_[ii].round(4))))
for j in range(2):print('best',j,sorted(opts.items(),key=lambda q:q[1][j])[:15]);
# A pure shared saturator, plus a shared slope, implies each affine correction is row proportional to its saturator mixing row.
print('C1 row slope ratios',np.sum(ops2[0][-1][1:5]*ops2[0][-1][5:],1)/np.sum(ops2[0][-1][5:]**2,1));
for nm,f in [('tanh',np.tanh),('atan',np.arctan),('erf',erf),('asinh',np.arcsinh),('softsign',lambda x:x/np.sqrt(1+x*x)),('sin',np.sin)]:
 def design(X,p): return aff(f(X*p[0])+p[1]*X)
 def res(p):
  E=design(U,p);return (E@fit(E,R)-R).ravel()
 o=min((least_squares(res,[k,0],bounds=([.1,-10],[8,10])) for k in (1.,2.,3.,5.)),key=lambda a:np.linalg.norm(a.fun));print('C1 shared',nm,o.x,np.std(o.fun))
