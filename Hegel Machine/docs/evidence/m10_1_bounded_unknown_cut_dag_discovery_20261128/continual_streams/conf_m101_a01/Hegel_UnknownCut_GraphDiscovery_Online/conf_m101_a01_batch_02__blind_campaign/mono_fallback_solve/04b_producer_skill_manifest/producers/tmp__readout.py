from explore import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor
np.set_printoptions(precision=3,suppress=True,linewidth=160)
Q=PolynomialFeatures(2); P=Q.fit_transform(x)
F=z@np.linalg.svd(z,full_matrices=False)[2][:2].T
print('telemetry',s['telemetry_vec'],'times',t,'A',A,'B',B)
for name,d,v in [('a',pa,pa['v']),('b',pb,pb['v']),('c1',pc,pc['v1']),('c2',pc,pc['v2'])]:
 e=tf(d['u'],aff(d,'v' if name in ('a','b') else 'v'+name[-1], name=='a'))-v;print(name,'u-range',d['u'].min(),d['u'].max(),'affine err',np.quantile(np.linalg.norm(e,axis=1),[0,.5,.9,1]))
dis=cdist(x,x,'sqeuclidean'); K=np.exp(-dis/(2*1.1**2)); KT=np.exp(-dis/(2*1.7**2));
def fitpred(K,p,target,tr,te,reg=.003):
 h=p[tr]; a=K[np.ix_(tr,tr)]+reg*np.eye(len(tr)); w=solve(a,np.c_[h,target[tr]],assume_a='pos'); b=np.linalg.lstsq(h.T@w[:,:h.shape[1]],h.T@w[:,h.shape[1]:],rcond=1e-10)[0]; return p[te]@b+K[np.ix_(te,tr)]@(w[:,h.shape[1]:]-w[:,:h.shape[1]]@b)
tr=np.arange(52); te=np.arange(52,72); yp=fitpred(K,P,y[:,None],tr,te)[:,0]; h=P@np.linalg.pinv(P); e=(y-h@y)/(1-h.diagonal()); print('rows i x y predicted residual ; all-row quadratic LOO residual');print(np.c_[te,x[te],y[te],yp,y[te]-yp,e[te]]); j=np.argsort(abs(e[:52]))[-10:]; print('train large raw residuals',np.c_[j,x[j],y[j],e[j]]);
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
res=[]
for tr,te in folds:
 basis=np.linalg.svd(z[tr],full_matrices=False)[2][:2].T; f=z@basis; fp=f.copy(); fp[te]=fitpred(KT,P,f,tr,te)
 def feat(q,kind):
  outer=(x[:,:,None]*q[:,None,:]).reshape(len(x),-1)
  if kind==0:return np.c_[P,q]
  if kind==1:return np.c_[P,q,outer,q*q,q[:,0]*q[:,1]]
  if kind==2:return np.c_[P,q,outer]
  if kind==3:return np.c_[P,q,(q[:,0,None]*P)[:,1:]]
  if kind==4:return np.c_[P,q,(q[:,1,None]*P)[:,1:]]
  return PolynomialFeatures(3).fit_transform(np.c_[x,q])
 result=[]
 for k in range(6):
  for reg in [1e-4,.02,.2,2.]:
   f0=feat(f,k); f1=feat(fp,k); mu=f0[tr].mean(0); sd=f0[tr].std(0); sd[sd<1e-8]=1; a=(f0[tr]-mu)/sd; b=np.linalg.solve(a.T@a+reg*np.eye(a.shape[1]),a.T@(y[tr]-y[tr].mean())); result.append([np.abs((f2[te]-mu)/sd@b+y[tr].mean()-y[te]).mean() for f2 in [f0,f1]])
 res.append(result)
r=np.array(res); print('conditional trajectory results: form, ridge, oracle/fitted 52, fitted worst, fitted mean');
for i in np.argsort(r[:,:,1].mean(0))[:12]:print(i//4,[1e-4,.02,.2,2.][i%4],r[0,i],r[:,i,1].max(),r[:,i,1].mean())
