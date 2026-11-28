exec(open('/tmp/readout7.py').read().split('folds=')[0])
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
rates=p.copy()
Yobs=np.c_[y,T];
for label,g in G.items():
 f=np.c_[aff(g),*[g[:,i]*g[:,j] for i in range(4) for j in range(i,4)]]; co=ls(f,Y)
 print('quadratic',label,'RMSE',np.std(f@co-Y,0));
 if label in ('DB','D','a'):
  print('coeff',co[5:].T.round(3));
print('corr',np.corrcoef(Y.T));print('y on amps rms',np.std(y-aff(q)@ls(aff(q),y)))
GD=np.c_[G['DB'],z]; folds=np.array_split(np.random.default_rng(202).permutation(len(z)),6)
def kernel(A,B,mode):
 R=cdist(A,B);return ((1+R+R*R/3)*np.exp(-R)) if mode=='mat' else np.exp(-R*R/2)
def one(A,B,Yt,mode):
 # Standardize axes using the training fold. Fit a mildly regularized length metric separately for each response.
 mu=A.mean(0);sd=A.std(0);A=(A-mu)/sd;B=(B-mu)/sd; m=Yt.mean();s=Yt.std();y=(Yt-m)/s;eye=np.eye(len(A)); nf=A.shape[1]
 dd=(A[:,None,:]-A[None,:,:])**2; de=(B[:,None,:]-A[None,:,:])**2
 def calc(p,d,grad=False):
  w=np.exp(-2*p[:nf]); R2=d@w; R=np.sqrt(R2); K=(1+R+R2/3)*np.exp(-R) if mode=='mat' else np.exp(-R2/2)
  if grad:return K, (1+R)*np.exp(-R)/3 if mode=='mat' else K, w
  return K
 def loss(p):
  K,g,w=calc(p,dd,True);reg=np.exp(p[-1]);C=cho_factor(K+reg*eye);b=cho_solve(C,y);inv=cho_solve(C,eye)-np.outer(b,b);a=y@b/len(y)
  # profile response variance
  v=(inv+np.outer(b,b)*(1-1/a))/2
  pen=.15*np.sum((p[:nf]-np.log(4))**2)
  return len(y)*np.log(a)/2+np.log(C[0].diagonal()).sum()+pen,np.r_[np.einsum('ij,ij,ijk,k->k',v,g,dd,w)+.3*(p[:nf]-np.log(4)),reg*np.trace(v)]
 p=minimize(loss,np.r_[np.full(nf,np.log(4)),-9.],method='L-BFGS-B',jac=True,bounds=[(-1,4)]*nf+[(-16,-3)],options={'maxiter':100}).x
 C=cho_factor(calc(p,dd)+np.exp(p[-1])*eye)
 return calc(p,de)@cho_solve(C,y)*s+m,p
for gn,g in [('D',G['D']),('DB',G['DB']),('DI',G['DI']),('a',z),('DB+a',GD)]:
 for mode in ('mat','rbf'):
  E=[];P=[]
  for hi in folds:
   tr=np.setdiff1d(np.arange(len(z)),hi); pred=[];p=[]
   for j in range(3):
    pp,pj=one(g[tr],g[hi],Y[tr,j],mode);pred.append(pp);p.append(pj)
   pred=np.asarray(pred).T; e=np.mean(abs(np.c_[pred[:,0],pred[:,1:]@F(rates).T]-Yobs[hi]),0)
   E.append(e);P.append(p)
  print(gn,mode,'mean',np.mean(E,0)[[0,1,4,8]],'worst',np.max(E,0)[[0,1,4,8]],'param',np.median(P,0).round(2))
