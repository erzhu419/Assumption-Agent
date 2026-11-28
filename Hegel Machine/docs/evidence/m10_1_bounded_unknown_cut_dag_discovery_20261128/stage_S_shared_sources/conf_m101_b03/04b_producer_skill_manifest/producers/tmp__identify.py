from modes import *
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.linear_model import ARDRegression
p=s['panel_c']; u=p['u']; V=np.c_[p['v1'],p['v2']]; P=aff(u); C=ls(P,V); R=V-P@C
np.set_printoptions(precision=4,suppress=True,linewidth=140)
print('B singular values',np.linalg.svd(B[1:],compute_uv=False)); print('C approx linear singular values',np.linalg.svd(C[1:,:4],compute_uv=False)); print('C raw std',V.std(0)); print('C cross-linear effects',ls(np.c_[P,V[:,4:]],V[:,:4])[5:]); print('C panel residual std',R.std(0))
for g in [u,aff(u)@B,(u-B[0])@np.linalg.inv(B[1:]), P@C[:,:4],u@C[1:,:4].T, V[:,:4]]:
 F=PolynomialFeatures(3).fit_transform(g); D=PolynomialFeatures(2).fit_transform(g); S=F[:,15:];
 print('C coordinate: sparse feature names and coefficients, max degree 3')
 for j in range(8):
  if j<4 and g is V[:,:4]:continue
  yy=V[:,j]; M=F if j<4 else D; m=M[:,1:].std(0); X=M[:,1:]/m; model=ARDRegression(fit_intercept=True).fit(X,yy); b=model.coef_/m; idx=np.argsort(abs(b))[-5:][::-1]; print(j,round(np.std(model.predict(X)-yy),4),[(PolynomialFeatures(3).fit(g).get_feature_names_out()[i+1],round(b[i],3)) for i in idx])
print('----- panel C LOOCV by kernel and spatial test -----')
for typ in ['poly','matern','rbf']:
 opts=[]
 for scale in [.03,.1,.3,1,3]:
  for d in ([2,3,4,5] if typ=='poly' else [1]):
   def kern(a,b):
    dd=cdist(a,b,'sqeuclidean')
    if typ=='poly':return (1+scale*a@b.T)**d
    if typ=='rbf':return np.exp(-scale*dd)+.5*(a@b.T)+1
    z=np.sqrt(scale*dd); return (1+z+z*z/3)*np.exp(-z)+.5*(a@b.T)+1
   K=kern(u,u)
   for a in [1e-5,.001,.01,.1]:
    I=solve(K+a*np.eye(len(u)),np.eye(len(u)),assume_a='pos'); E=np.mean(abs(I@V/np.diag(I)[:,None]),0); opts.append((E.mean(),E,scale,d,a))
 print(typ, min(opts,key=lambda r:r[0]))
print('---- GP likelihood selected kernels ----')
for X in [x,q,base['aib']]:
 for j in [0,1,2]:
  for mode in ['rbf','matern','poly','rat']:
   def kernel(a,b,h):
    W=np.exp(h[:4]); da=a*W; db=b*W; dist=cdist(da,db,'sqeuclidean'); dot=a@b.T
    if mode=='rbf':g=np.exp(-dist/2)
    if mode=='matern':
     v=np.sqrt(5*dist); g=(1+v+v*v/3)*np.exp(-v)
    if mode=='rat':g=(1+dist/(2*np.exp(h[8])))**(-np.exp(h[8]))
    if mode=='poly':g=(1+da@db.T)**4
    return np.exp(h[4])*g+np.exp(h[5])*(1+dot)+np.exp(h[6])*(1+dot)**2
   a=X[:n]; yy=Z[:n,j]; yy=(yy-yy.mean())/yy.std()
   def loss(h,ret=False):
    K=kernel(a,a,h); K.flat[::n+1]+=np.exp(h[7]); cf=cho_factor(K); b=cho_solve(cf,yy)
    if ret:return kernel(X[n:],a,h)@b*Z[:n,j].std()+Z[:n,j].mean()
    return .5*(yy@b)+np.log(np.diag(cf[0])).sum()
   h=np.r_[np.zeros(4)-.7,0.,-1.,-3.,-8.,1.]; fit=minimize(loss,h,method='L-BFGS-B',bounds=[(-5,3)]*4+[(-12,7)]*3+[(-15,-.5),(-4,5)],options={'maxiter':140}); P=loss(fit.x,True); print('gp',j,mode,round(np.mean(abs(P-Z[n:,j])),5),np.round(fit.x,2))
